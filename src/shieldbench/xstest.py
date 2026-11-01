"""Over-refusal on a public benchmark: XSTest's 250 safe-but-scary-sounding prompts.
   python -m shieldbench.xstest --guards g1,g2,g3,g4 --export results/xstest.csv
A guard that blocks one of these has blocked a harmless request. Guards here are made to catch prompt injection,
not violent words, so a block is a false alarm. Only the guard is run (no bot), so it is cheap."""
import argparse
import pandas as pd
from . import config
from .eval import metrics
from .eval.cost import CostTracker
from .guardrails.pipeline import make_guard
from .store.db import Store

CSV = config.ROOT / "data" / "xstest" / "xstest_prompts.csv"


def load_safe(path=None) -> pd.DataFrame:
    d = pd.read_csv(path or CSV)
    return d[d["label"] == "safe"].reset_index(drop=True)


def run(guards, store, budget=1.0, limit=None, path=None):
    safe = load_safe(path)
    if limit:
        safe = safe.head(limit)
    tracker = CostTracker(budget)
    rows = []
    for gk in guards:
        g = make_guard(gk, store=store, tracker=tracker)
        for r in safe.itertuples():
            v = g.check_input(r.prompt)
            rows.append({"guard": g.name, "id": r.id, "type": r.type, "prompt": r.prompt, "blocked": int(bool(v.blocked)), "reason": v.reason})
    return pd.DataFrame(rows), tracker.spent


def summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for g, grp in df.groupby("guard"):
        k, n = int(grp.blocked.sum()), len(grp)
        p, lo, hi = metrics.wilson(k, n)
        out.append({"guard": g, "safe prompts": n, "blocked": k, "blocked %": round(p * 100, 1), "interval %": f"{lo*100:.1f} to {hi*100:.1f}"})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guards", default="g1,g2,g3,g4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--export", default="")
    a = ap.parse_args()
    df, spent = run(a.guards.split(","), Store(a.db), a.budget, a.limit)
    print(f"{df.id.nunique()} safe XSTest prompts, spent ${spent:.4f}")
    print(summary(df).to_string(index=False))
    if a.export:
        df.to_csv(a.export, index=False)
        print("wrote", a.export)


if __name__ == "__main__":
    main()
