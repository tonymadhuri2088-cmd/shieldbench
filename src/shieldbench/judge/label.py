"""Human labelling tools (checklist C6).
  export : python -m shieldbench.judge.label export --n 100          -> data/labels/label_sheet.csv + key.csv
  kappa  : python -m shieldbench.judge.label kappa --a A.csv --b B.csv
Labelers see NO judge verdicts (blind). 1 = attack succeeded, 0 = it failed."""
import argparse
import random
import sqlite3
from pathlib import Path
import pandas as pd
from .. import config
from ..attacks.mutators import resolve
from .claude_judge import RUBRICS


def cohen_kappa(a, b) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in set(a) | set(b))
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def export(db, n, seed, out_dir, run=None):
    con = sqlite3.connect(db)
    q = "SELECT attack_id, category, guard, response, success, judge_llm FROM results WHERE benign=0 AND blocked=0"
    args = ()
    if run:
        q, args = q + " AND run_id=?", (run,)
    df = pd.read_sql_query(q, con, params=args).drop_duplicates(["attack_id", "response"])
    rnd = random.Random(seed)
    groups = {c: g.sample(frac=1, random_state=seed) for c, g in df.groupby("category")}
    picked, i = [], 0
    while len(picked) < min(n, len(df)):           # round-robin => stratified by category
        for c in sorted(groups):
            if i < len(groups[c]) and len(picked) < n:
                picked.append(groups[c].iloc[i])
        i += 1
    rnd.shuffle(picked)
    sheet, key = [], []
    for k, r in enumerate(picked, 1):
        a = resolve(r.attack_id)
        iid = f"L{k:03d}"
        sheet.append(dict(item_id=iid, category=r.category, success_means=RUBRICS[r.category],
                          attack=a.text + (f"  [DOCUMENT: {a.document}]" if a.document else ""),
                          bot_response=r.response, label=""))
        key.append(dict(item_id=iid, attack_id=r.attack_id, category=r.category, rule_judge=r.success, claude_judge=r.judge_llm))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sheet).to_csv(out / "label_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    print(f"wrote {len(sheet)} items to {out}/label_sheet.csv (give a COPY to each labeler) and key.csv (keep private)")


def kappa_report(a_path, b_path, key_path):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    m = a[["item_id", "label"]].merge(b[["item_id", "label"]], on="item_id", suffixes=("_a", "_b")).dropna()
    la, lb = m.label_a.astype(int).tolist(), m.label_b.astype(int).tolist()
    print(f"labeled by both: {len(m)} items")
    print(f"human vs human   kappa = {cohen_kappa(la, lb):.3f}  (agreement {sum(x == y for x, y in zip(la, lb)) / len(m):.1%})")
    agreed = m[m.label_a == m.label_b]
    key = pd.read_csv(key_path).merge(agreed[["item_id", "label_a"]], on="item_id")
    key = key.rename(columns={"label_a": "human"})
    print(f"{len(m) - len(agreed)} items where humans disagreed: settle these by discussion before trusting the numbers")
    for name in ("rule_judge", "claude_judge"):
        k = key.dropna(subset=[name])
        k = k[k[name].isin([0, 1])]
        if k.empty:
            print(f"{name}: no data (run the judge first)")
            continue
        h, j = k.human.astype(int).tolist(), k[name].astype(int).tolist()
        fp = sum(x == 0 and y == 1 for x, y in zip(h, j))
        fn = sum(x == 1 and y == 0 for x, y in zip(h, j))
        print(f"{name:12s} vs human  kappa = {cohen_kappa(h, j):.3f}  agree {sum(x == y for x, y in zip(h, j))}/{len(h)}  false-pos {fp}  false-neg {fn}")


def check_report(labels_path, key_path, sheet_path, out_path=None):
    """One human labeler vs the two judges (use when no second labeler exists)."""
    h = pd.read_csv(labels_path)[["item_id", "label"]].dropna()
    key = pd.read_csv(key_path).merge(h, on="item_id")
    sheet = pd.read_csv(sheet_path)[["item_id", "bot_response"]]
    key = key.merge(sheet, on="item_id", how="left")
    key["label"] = key.label.astype(int)
    print(f"items labeled by you: {len(key)}   you said 'attack succeeded' on {int(key.label.sum())} ({key.label.mean():.0%})")
    rows = []
    for name in ("rule_judge", "claude_judge"):
        k = key[key[name].isin([0, 1])]
        hh, jj = k.label.tolist(), k[name].astype(int).tolist()
        fp = sum(x == 0 and y == 1 for x, y in zip(hh, jj))
        fn = sum(x == 1 and y == 0 for x, y in zip(hh, jj))
        agree = sum(x == y for x, y in zip(hh, jj))
        print(f"{name:12s} vs you: agree {agree}/{len(k)} ({agree/len(k):.0%})  kappa {cohen_kappa(hh, jj):.3f}  "
              f"false alarms {fp} (judge said success, you said no)  misses {fn} (judge said fail, you said success)")
        rows.append((name, agree, len(k), fp, fn))
    bad = key[(key.label != key.claude_judge.astype(float)) | (key.label != key.rule_judge.astype(float))].copy()
    print(f"\n{len(bad)} items where you and at least one judge disagree. Re-read these in the labeling tool:")
    for _, r in bad.iterrows():
        txt = str(r.bot_response).replace("\n", " ")[:90]
        print(f"  {r.item_id}  {r.category:18s} you={r.label}  claude={r.claude_judge}  rule={r.rule_judge}  | {txt}")
    if out_path:
        bad[["item_id", "category", "label", "claude_judge", "rule_judge"]].to_csv(out_path, index=False)
        print(f"\nsaved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("--labels", required=True)
    c.add_argument("--key", default=str(config.ROOT / "data" / "labels" / "key.csv"))
    c.add_argument("--sheet", default=str(config.ROOT / "data" / "labels" / "label_sheet.csv"))
    c.add_argument("--out", default=None)
    e = sp.add_parser("export")
    e.add_argument("--db", default=str(config.DB_PATH))
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--seed", type=int, default=config.SEED)
    e.add_argument("--run", default=None)
    e.add_argument("--out", default=str(config.ROOT / "data" / "labels"))
    k = sp.add_parser("kappa")
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    k.add_argument("--key", default=str(config.ROOT / "data" / "labels" / "key.csv"))
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.db, a.n, a.seed, a.out, a.run)
    elif a.cmd == "check":
        check_report(a.labels, a.key, a.sheet, a.out)
    else:
        kappa_report(a.a, a.b, a.key)


if __name__ == "__main__":
    main()
