"""Independent attacker and judge: take Claude out of two of its four jobs.

Until now Claude wrote the attack variants, guarded, and judged. This module lets a model from ANOTHER family (anything that speaks the
OpenAI chat format, e.g. Ollama's gemma2 or llama3.2) do two of those jobs, so the results do not rest on one family's blind spots.

  1. attacks : a local model rewrites the 22 seed attacks, many per seed (a bigger set, written by someone other than Claude)
       python -m shieldbench.independent attacks --model gemma2:2b --per-seed 10 --out data/generated/variants_local.json
     then scan any bot with it:  python -m shieldbench.byobot --models gemma2:2b,llama3.2:3b --attack-file data/generated/variants_local.json --guards g1,g2
  2. judge   : a local model re-judges the replies that Claude and the rule judge already judged; we report how often they agree
       python -m shieldbench.independent judge --db results/final.db --model gemma2:2b --limit 100
  3. ceiling : how small a rate "zero worked" can hide, for different numbers of attacks
       python -m shieldbench.independent ceiling

Honest limits: a 2B model writes weaker attacks and judges worse than Claude, so low agreement mostly says "small judges are unreliable", and the
real fix is a strong judge from another family. The numbers are reported as they come out."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import config
from .attacks.generate import generate_variants, save
from .attacks.library import ATTACKS, with_
from .eval.metrics import wilson
from .judge.claude_judge import SYSTEM as JUDGE_SYSTEM, build_prompt, parse
from .judge.label import cohen_kappa


def make_ask(url: str, model: str, key: str = "", temperature: float = 0.0, max_tokens: int = 700, timeout: int = 300):
    """ask(system, user) -> text, for any OpenAI-format server."""
    import requests

    def ask(system: str, user: str) -> str:
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + key
        r = requests.post(url.rstrip("/") + "/chat/completions", headers=h, timeout=timeout, json={
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    return ask


def local_attacks(ask, per_seed: int = 10, prefix: str = "loc"):
    """Variants of the seed attacks written by `ask` (a non-Claude model). Ids start with `prefix-` so they never clash with Claude's."""
    got = generate_variants(ATTACKS, ask, per_seed)
    return [with_(a, id=a.id.replace("gen-", prefix + "-", 1), family=a.family.replace("gen_", prefix + "_", 1)) for a in got]


def judge_rows(rows, ask):
    """rows: dicts with attack (Attack), response, claude (0/1), rule (0/1). Returns the agreement report."""
    out, bad = [], 0
    for r in rows:
        v = parse(ask(JUDGE_SYSTEM, build_prompt(r["attack"], r["response"])))
        if v is None:
            bad += 1
            continue
        out.append(dict(local=int(v[0]), claude=int(r["claude"]), rule=int(r["rule"]), id=r["attack"].id, category=r["attack"].category, response=" ".join(str(r["response"]).split())[:140]))
    rep = {"n": len(out), "unparseable": bad}
    for name in ("claude", "rule"):
        a, b = [x["local"] for x in out], [x[name] for x in out]
        agree = sum(x == y for x, y in zip(a, b))
        rep[name] = {"agree": agree, "kappa": round(cohen_kappa(a, b), 3) if out else None,
                     "local_says_success_judge_says_no": sum(x == 1 and y == 0 for x, y in zip(a, b)),
                     "local_says_fail_judge_says_success": sum(x == 0 and y == 1 for x, y in zip(a, b))}
    rep["disagreements"] = [x for x in out if x["local"] != x["claude"]][:10]
    return rep


def zero_ceiling(ns=(20, 40, 124, 300, 500, 1000)):
    """If 0 of n attacks worked, the true rate could still be as high as this (95% Wilson upper bound), in percent."""
    return [{"attacks": n, "upper_percent": round(100 * wilson(0, n)[2], 1)} for n in ns]


def main(argv=None, ask=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("attacks", "judge"):
        p = sp.add_parser(name)
        p.add_argument("--url", default="http://localhost:11434/v1")
        p.add_argument("--model", required=True)
        p.add_argument("--key", default="")
    sp.choices["attacks"].add_argument("--per-seed", type=int, default=10)
    sp.choices["attacks"].add_argument("--out", default=str(config.ROOT / "data" / "generated" / "variants_local.json"))
    sp.choices["judge"].add_argument("--db", required=True)
    sp.choices["judge"].add_argument("--limit", type=int, default=100)
    sp.choices["judge"].add_argument("--out", default="results/independent_judge.json")
    sp.add_parser("ceiling")
    a = ap.parse_args(argv)
    if a.cmd == "ceiling":
        print("If this many attacks were run and NONE worked, the true rate could still be as high as:")
        for r in zero_ceiling():
            print(f"  {r['attacks']:5d} attacks -> up to {r['upper_percent']}%")
        return 0
    if a.cmd == "attacks":
        ask = ask or make_ask(a.url, a.model, a.key, temperature=0.9)
        v = local_attacks(ask, a.per_seed)
        save(v, a.out)
        by = {}
        for x in v:
            by[x.category] = by.get(x.category, 0) + 1
        print(f"wrote {len(v)} attacks written by {a.model} to {a.out}; per category: {by}")
        return len(v)
    from .attacks.mutators import resolve
    from .store.db import Store
    con = Store(a.db).con
    q = ("SELECT attack_id, response, judge_llm, success FROM results WHERE benign=0 AND blocked=0 AND judge_llm IN (0,1) "
         "AND response IS NOT NULL GROUP BY attack_id, response ORDER BY attack_id LIMIT ?")
    rows = []
    for aid, resp, jl, sc in con.execute(q, (a.limit,)):
        try:
            rows.append(dict(attack=resolve(aid), response=resp, claude=jl, rule=sc))
        except KeyError:
            continue
    rep = judge_rows(rows, ask or make_ask(a.url, a.model, a.key, max_tokens=300))
    rep["judge_model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{a.model} judged {rep['n']} replies ({rep['unparseable']} unreadable)")
    for name in ("claude", "rule"):
        r = rep[name]
        print(f"  vs {name:6s} judge: agree {r['agree']}/{rep['n']}  kappa {r['kappa']}  (says success, they say no: {r['local_says_success_judge_says_no']};  says fail, they say success: {r['local_says_fail_judge_says_success']})")
    print(f"saved {a.out}")
    return rep["n"]


if __name__ == "__main__":
    main()
