"""One-command automation. Runs the test suites in order: attack run -> Claude judge -> report -> datasheet -> regression gate.
Each step is an ordinary ShieldBench command; the plan is printed first, finished suites are skipped (safe to re-run), and one
failed step does not stop the others. A summary file is written at the end.

    python -m shieldbench.autorun --dry-run                         show the plan, run nothing
    python -m shieldbench.autorun --suites heldout4 --target ollama
    python -m shieldbench.autorun --suites all --target ollama       every untouched set (costs a few dollars of Claude judging at most)
    python -m shieldbench.autorun --suites heldout4 --force          run again even if the results file exists

Do not press Ctrl+C during a run (the cache lets you resume by running the same command again)."""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

SUITES = {  # name: attack-set flag
    "heldout2": "heldout2",
    "heldout3": "heldout3",
    "heldout4": "heldout4",
    "generated": "generated",
    "main": "main",
}
GUARDS = "g0,g1,g2,g3,g4"


def plan(suites: list[str], target: str, out: str = "results", guards: str = GUARDS, force: bool = False) -> list[dict]:
    steps = []
    for s in suites:
        if s not in SUITES:
            raise SystemExit(f"unknown suite {s!r}; choose from {sorted(SUITES)} or 'all'")
        tag = s if target == "ollama" else f"{s}_{target}"
        db, raw, judged = f"{out}/{tag}.db", f"{out}/{tag}_raw.csv", f"{out}/{tag}_judged.csv"
        py = [sys.executable, "-m"]
        extra = ["--hard-benign"] if s == "main" else []
        steps.append({"suite": s, "judged": judged, "skip": (Path(judged).exists() and not force), "cmds": [
            py + ["shieldbench.run", "--target", target, "--guards", guards, "--attack-set", SUITES[s], "--db", db, "--export", raw] + extra,
            py + ["shieldbench.judge.run_judge", "--db", db],
            py + ["shieldbench.report", "--db", db, "--csv", judged],
        ]})
    return steps


def finish_cmds(out: str = "results", gate_csv: str | None = None) -> list[list[str]]:
    py = [sys.executable, "-m"]
    cmds = [py + ["shieldbench.datasheet", "--csv", gate_csv or f"{out}/final_judged.csv", "--out", f"{out}/datasheet.md"]]
    cmds.append(py + ["shieldbench.gate"] + (["--csv", gate_csv] if gate_csv else []))
    return cmds


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run ShieldBench suites end to end")
    ap.add_argument("--suites", default="heldout4", help="comma list or 'all': " + ",".join(SUITES))
    ap.add_argument("--target", default="ollama", choices=["ollama", "claude", "mock"])
    ap.add_argument("--guards", default=GUARDS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    suites = [s for s in SUITES if s != "main"] if a.suites == "all" else [s.strip() for s in a.suites.split(",") if s.strip()]
    steps = plan(suites, a.target, a.out, a.guards, a.force)
    print(f"ShieldBench autorun: {len(steps)} suite(s), target {a.target}, guards {a.guards}")
    for st in steps:
        print(f"  - {st['suite']}: {'skip (results exist, use --force)' if st['skip'] else 'run'}")
    if a.dry_run:
        for st in steps:
            for c in st["cmds"]:
                print("   would run:", " ".join(c))
        for c in finish_cmds(a.out):
            print("   would run:", " ".join(c))
        return 0
    t0, summary = time.time(), []
    for st in steps:
        if st["skip"]:
            summary.append((st["suite"], "skipped (already done)"))
            continue
        ok = True
        for c in st["cmds"]:
            if _run(c) != 0:
                ok = False
                break
        summary.append((st["suite"], "done" if ok else "FAILED (run the same command again; the cache resumes)"))
    gate_rc = [_run(c) for c in finish_cmds(a.out)]
    lines = [f"# ShieldBench autorun summary ({time.strftime('%Y-%m-%d %H:%M')})", ""] + [f"- {s}: {r}" for s, r in summary]
    lines.append(f"- datasheet: {'ok' if gate_rc[0] == 0 else 'failed'}; regression gate: {'PASS' if gate_rc[1] == 0 else 'FAIL (exit %d)' % gate_rc[1]}")
    lines.append(f"- took {time.time() - t0:.0f} s")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "autorun_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if all(r in ("done",) or r.startswith("skipped") for _, r in summary) and gate_rc[1] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
