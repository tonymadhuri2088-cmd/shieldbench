"""Print every results table for a run:  python -m shieldbench.report [--run RUN_ID]"""
import argparse
import pandas as pd
from . import config
from .eval import metrics
from .store.db import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--run", default="latest")
    ap.add_argument("--csv", default="", help="also save this run (with judge columns) to a CSV")
    a = ap.parse_args()
    store = Store(a.db)
    run = a.run
    if run == "latest":
        # every guard's most recent run, merged, so guards run at different times appear together
        allr = store.results_df()
        last = allr.sort_values("ts").groupby("guard").run_id.last()
        df = allr[allr.run_id == allr.guard.map(last)].copy()
        run = ", ".join(sorted(set(last.values)))
    else:
        df = store.results_df(run)
    df = metrics.with_verdict(df)
    pd.set_option("display.width", 200)
    judged = int(df["judge_llm"].isin([0, 1]).sum()) if "judge_llm" in df else 0
    print(f"RUN {run}   target(s): {', '.join(df.target.unique())}   rows: {len(df)}   judged by Claude: {judged}")
    print("\n== Attack success rate by guard and category (final verdict, 95% Wilson) ==")
    t = metrics.asr_table(df, "verdict")
    for c in ("asr", "ci_low", "ci_high"):
        t[c] = (t[c] * 100).round(1)
    print(t.to_string(index=False))
    from .taxonomy import risk_table
    print("\n== Risk by category and severity (attack success %, with OWASP LLM Top 10 item) ==")
    print(risk_table(df).to_string(index=False))
    print("\n== Overall ASR per guard: rule judge vs final verdict ==")
    print(metrics.judge_comparison(df).round(3).to_string(index=False))
    print("\n== ASR % by mutation and guard (final verdict) ==")
    att = df[df["benign"] == 0]
    print((att.groupby(["mutation", "guard"]).verdict.mean().unstack() * 100).round(1).to_string())
    print("\n== How each guard produced its blocks (attacks only) ==")
    print(metrics.block_breakdown(df).to_string(index=False))
    print("\n== Over-refusal on benign prompts ==")
    print(metrics.overrefusal_table(df[df["category"] != "benign_hard"]).round(3).to_string(index=False))
    if (df["category"] == "benign_hard").any():
        print("\n== Over-refusal on HARD benign prompts (tricky but harmless) ==")
        print(metrics.overrefusal_table(df[df["category"] == "benign_hard"]).round(3).to_string(index=False))
    print("\n== Task success on clean documents ==")
    print(metrics.utility_table(df).round(3).to_string(index=False))
    print("\n== Latency: model (s) and guard overhead (ms) ==")
    print(metrics.latency_table(df).round(3).to_string(index=False))
    print("\n== How often the Claude guard was used (guard time above 0.2 s) ==")
    print(metrics.claude_share(df).to_string(index=False))
    ep = store.episodes_df()
    if len(ep):
        last = ep.sort_values("ts").run_id.iloc[-1]
        print(f"\n== Adaptive attacker (latest adaptive run {last}) ==")
        print(metrics.adaptive_table(ep[ep.run_id == last]).round(3).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print("\nsaved", a.csv)


if __name__ == "__main__":
    main()
