"""Improve the G2 detector for real traffic WITHOUT touching the one used in your results.

    python -m shieldbench.improve --budget 0.5                  # writes data/models/g2_v2.joblib, prints old versus new, changes nothing else
    python -m shieldbench.improve --adopt                       # same, then copies v2 over g2.joblib (a backup is kept) ONLY if the check passes

What it adds to the training data (all of it passes through the same overlap filter as before, so nothing resembling a test prompt is used):
  1. Attacks and harmless messages written in native scripts (Hindi, Tamil, Telugu, Bengali, Marathi, Kannada) by Claude Haiku (costs cents, key needed).
  2. Your own corrections from results/feedback.jsonl (the Live page's 'wrongly blocked' / 'this was an attack' buttons).
A quarter of the new native-script rows are held OUT of training and used to compare old and new honestly.

The check before adopting: on the five fixed test sets the new detector must not catch more than 2 fewer attacks in any set and must not block more harmless
messages in total; on the held-out native-script rows it must catch more attacks. If it fails, nothing changes. After adopting, rerun the evaluation
(autorun) before quoting any headline number, because the guard that produced them is no longer the same."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .arena import regression, scores, THRESHOLD
from .guardrails.detector_guard import MODEL_PATH, fit_model, build_pipeline, expand
from .guardrails.train import _pick_thresholds
from .guardrails.train_data import ATTACK_KINDS, _ask, drop_overlap, _norm

LANGS = {"hi": "Hindi (Devanagari script)", "ta": "Tamil (Tamil script)", "te": "Telugu (Telugu script)",
         "bn": "Bengali (Bengali script)", "mr": "Marathi (Devanagari script)", "kn": "Kannada (Kannada script)"}


def gen_multilingual(ask, langs: list[str], per_call: int = 12) -> pd.DataFrame:
    """ask(name, prompt) -> (items, cost, refused). Returns rows text,label,category,style,source."""
    rows, cost = [], 0.0
    for lg in langs:
        name = LANGS[lg]
        for cat, desc in ATTACK_KINDS.items():
            items, c, refused = ask(f"ml_atk_{lg}_{cat}", f"Write {per_call} different user messages written ENTIRELY in {name}, not transliterated: {desc}. Vary the wording a lot.")
            cost += c
            print(f"  {lg} {cat:20s} {len(items):3d} items" + ("  (declined)" if refused or not items else ""))
            rows += [dict(text=t, label=1, category=cat, style=lg, source="claude_ml") for t in items]
        items, c, refused = ask(f"ml_ben_{lg}", f"Write {per_call + 8} different, completely harmless customer messages to a shop support bot, written ENTIRELY in {name}, not transliterated: order status, refunds, product questions, shipping, own account help. Vary length and wording.")
        cost += c
        print(f"  {lg} {'harmless':20s} {len(items):3d} items" + ("  (declined)" if refused or not items else ""))
        rows += [dict(text=t, label=0, category="benign", style=lg, source="claude_ml") for t in items]
    return pd.DataFrame(rows, columns=["text", "label", "category", "style", "source"]), cost


def load_feedback(path) -> pd.DataFrame:
    p = Path(path)
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
                rows.append(dict(text=str(d["text"]), label=int(d["label"]), category="feedback", style="feedback", source="feedback"))
            except (ValueError, KeyError):
                continue
    return pd.DataFrame(rows, columns=["text", "label", "category", "style", "source"])


def split_ml(ml: pd.DataFrame, frac: float = 0.25, seed: int = config.SEED):
    """Hold a share of the native-script rows out of training; returns (train_part, held_out)."""
    if ml.empty:
        return ml, ml
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(ml)) < frac
    return ml[~mask].reset_index(drop=True), ml[mask].reset_index(drop=True)


def compare(old, new, sets: dict, ml_test: pd.DataFrame) -> dict:
    o, n = regression(old, sets), regression(new, sets)
    rows = []
    for a, b in zip(o, n):
        rows.append(dict(set=a["set"], attacks=a["attacks"], caught_old=a["caught"], caught_new=b["caught"], harmless=a["harmless"], blocked_old=a["blocked"], blocked_new=b["blocked"]))
    ml = {}
    if not ml_test.empty:
        for tag, m in (("old", old), ("new", new)):
            sc = scores(m, list(ml_test.text))
            y = ml_test.label.to_numpy()
            ml[tag] = dict(attacks=int((y == 1).sum()), caught=int(((sc >= THRESHOLD) & (y == 1)).sum()), harmless=int((y == 0).sum()), blocked=int(((sc >= THRESHOLD) & (y == 0)).sum()))
    return {"sets": rows, "native": ml}


def verdict(cmp: dict) -> tuple[bool, list[str]]:
    why = []
    for r in cmp["sets"]:
        if r["caught_new"] < r["caught_old"] - 2:
            why.append(f"{r['set']}: catches {r['caught_old'] - r['caught_new']} fewer attacks")
    if sum(r["blocked_new"] for r in cmp["sets"]) > sum(r["blocked_old"] for r in cmp["sets"]):
        why.append("blocks more harmless messages in total on the fixed sets")
    nat = cmp["native"]
    if nat and nat["new"]["caught"] <= nat["old"]["caught"]:
        why.append("does not catch more native-script attacks than the old detector")
    if not nat:
        why.append("no native-script rows were available to test the main improvement")
    return (not why), why


def main(argv=None, ask=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.ROOT / "data" / "train" / "train.csv"))
    ap.add_argument("--feedback", default="results/feedback.jsonl")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--out", default=str(MODEL_PATH.with_name("g2_v2.joblib")))
    ap.add_argument("--adopt", action="store_true", help="copy v2 over g2.joblib if the check passes (a backup is kept)")
    ap.add_argument("--C", type=float, default=4.0)
    a = ap.parse_args(argv)
    import joblib
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    langs = [x.strip() for x in a.langs.split(",") if x.strip() in LANGS]
    spent = 0.0
    ml = pd.DataFrame(columns=["text", "label", "category", "style", "source"])
    if langs:
        if ask is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                from dotenv import load_dotenv
                load_dotenv(config.ROOT / ".env")
            import anthropic
            client = anthropic.Anthropic()
            raw = Path(a.data).parent / "raw_ml"
            raw.mkdir(parents=True, exist_ok=True)
            state = {"spent": 0.0}

            def ask(name, prompt):
                if state["spent"] > a.budget:
                    raise SystemExit(f"budget ${a.budget} reached")
                items, cost, refused = _ask(client, config.CLAUDE_TARGET_MODEL, prompt, raw, name)
                state["spent"] += cost
                return items, cost, refused
        print("Writing native-script examples:")
        ml, spent = gen_multilingual(ask, langs, a.per_call)
    ml = ml.drop_duplicates("text").reset_index(drop=True)
    ml, dropped_ml = drop_overlap(ml) if len(ml) else (ml, 0)
    ml_train, ml_test = split_ml(ml)
    fb = load_feedback(a.feedback)
    base = pd.read_csv(a.data).dropna(subset=["text"]).reset_index(drop=True)
    df = pd.concat([base, ml_train, fb], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    test_norm = {_norm(t) for t in ml_test.text}
    df = df[~df.text.map(_norm).isin(test_norm)].reset_index(drop=True)          # the held-out rows must never be trained on
    y = df.label.to_numpy()
    print(f"\ntraining rows: {len(df)} (base {len(base)}, native-script {len(ml_train)}, your feedback {len(fb)}); held out for testing: {len(ml_test)}; dropped as too close to test prompts: {dropped_ml}")
    X = np.array([expand(t) for t in df.text], dtype=object)
    oof = cross_val_predict(build_pipeline(a.C), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method="predict_proba")[:, 1]
    hi, lo = _pick_thresholds(y, oof)
    model = fit_model(df.text, y, a.C)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thr": 0.5, "hi": hi, "lo": lo, "meta": {"rows": int(len(df)), "attacks": int(y.sum()), "C": a.C, "seed": config.SEED, "native_rows": int(len(ml_train)), "feedback_rows": int(len(fb))}}, a.out)
    old = joblib.load(MODEL_PATH)["model"]
    from .guardrails.detector_eval import test_sets
    cmp = compare(old, model, test_sets(), ml_test)
    print(f"\n{'set':40s}{'attacks caught old -> new':>28s}{'harmless blocked old -> new':>30s}")
    for r in cmp["sets"]:
        print(f"{r['set']:40s}{str(r['caught_old']) + ' -> ' + str(r['caught_new']) + '  of ' + str(r['attacks']):>28s}{str(r['blocked_old']) + ' -> ' + str(r['blocked_new']) + '  of ' + str(r['harmless']):>30s}")
    for tag in ("old", "new"):
        if tag in cmp["native"]:
            n = cmp["native"][tag]
            print(f"{'native-script held-out, ' + tag:40s}{str(n['caught']) + ' of ' + str(n['attacks']):>28s}{str(n['blocked']) + ' of ' + str(n['harmless']):>30s}")
    ok, why = verdict(cmp)
    print(f"\nsaved {a.out}. Cost ${spent:.3f}. Check: " + ("PASSED, v2 is at least as good on the fixed sets and better on native scripts." if ok else "NOT passed: " + "; ".join(why)))
    if a.adopt:
        if ok:
            shutil.copy(MODEL_PATH, MODEL_PATH.with_name("g2_backup.joblib"))
            shutil.copy(a.out, MODEL_PATH)
            print("adopted: g2.joblib replaced (backup: g2_backup.joblib). Rerun the evaluation before quoting headline numbers.")
        else:
            print("not adopted; g2.joblib unchanged.")
    return ok


if __name__ == "__main__":
    main()
