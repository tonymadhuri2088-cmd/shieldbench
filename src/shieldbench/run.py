"""CLI:  python -m shieldbench.run --target mock --guards g0,g1 --mutate base64,roleplay"""
import argparse
from . import config
from .eval.runner import run
from .eval import metrics
from .store.db import Store


def make_target(name: str):
    if name == "mock":
        from .targets.mock_target import MockTarget
        return MockTarget()
    if name == "ollama":
        from .targets.ollama_target import OllamaTarget
        return OllamaTarget()
    if name == "openai":
        from .targets.openai_target import OpenAICompatTarget
        return OpenAICompatTarget()
    if name == "claude":
        from .targets.claude_target import ClaudeTarget
        return ClaudeTarget()
    raise SystemExit(f"unknown target {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mock", choices=["mock", "ollama", "claude", "openai"], help="openai = any OpenAI-compatible server, set OPENAI_COMPAT_URL / _MODEL / _KEY")
    ap.add_argument("--guards", default="g0,g1")
    ap.add_argument("--mutate", default="", help="comma list: base64,leetspeak,roleplay,hinglish,multi_turn")
    ap.add_argument("--hard-benign", action="store_true", help="add 36 tricky-but-harmless prompts for over-refusal")
    ap.add_argument("--attack-set", default="main", choices=["main", "heldout", "heldout2", "heldout3", "heldout4", "generated"], help="heldout/heldout2/heldout3/heldout4 = attacks written after the G2 detector was trained")
    ap.add_argument("--guard-context", default="last", choices=["last", "full"], help="last = guards read only the newest message; full = the whole conversation")
    ap.add_argument("--limit", type=int, default=None, help="small batch first!")
    ap.add_argument("--budget", type=float, default=config.BUDGET_USD)
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--export", default="", help="also write a CSV of this run to this path")
    a = ap.parse_args()
    store = Store(a.db)
    rid, spent = run(make_target(a.target), store, guards=a.guards.split(","),
                     mutations=[m for m in a.mutate.split(",") if m], budget_usd=a.budget, limit=a.limit,
                     hard_benign=a.hard_benign, attack_set=a.attack_set, guard_context=a.guard_context)
    df = store.results_df(rid)
    print(f"run {rid}: {len(df)} rows, spent ${spent:.4f}")
    print(metrics.asr_table(df).round(3).to_string(index=False))
    print(metrics.overrefusal_table(df).round(3).to_string(index=False))
    print(metrics.utility_table(df).round(3).to_string(index=False))
    if a.export:
        df.to_csv(a.export, index=False)
        print("wrote", a.export)


if __name__ == "__main__":
    main()
