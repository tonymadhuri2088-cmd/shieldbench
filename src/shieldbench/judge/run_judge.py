"""Judge stored results with Claude:  python -m shieldbench.judge.run_judge --limit 10"""
import argparse
from .. import config
from ..attacks.mutators import resolve
from ..eval.cost import CostTracker
from ..store.db import Store
from .claude_judge import ClaudeJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--run", default="latest")
    ap.add_argument("--limit", type=int, default=None, help="small batch first!")
    ap.add_argument("--budget", type=float, default=config.BUDGET_USD)
    a = ap.parse_args()
    store = Store(a.db)
    con = store.con
    run = a.run
    if run == "latest":
        run = con.execute("SELECT run_id FROM results ORDER BY ts DESC LIMIT 1").fetchone()[0]
    tracker = CostTracker(a.budget)
    judge = ClaudeJudge(store, tracker=tracker)
    # blocked requests can never succeed: no API call needed
    con.execute("UPDATE results SET judge_llm=0, judge_llm_reason='blocked by guard' "
                "WHERE run_id=? AND benign=0 AND blocked=1 AND judge_llm IS NULL", (run,))
    rows = con.execute("SELECT rowid, attack_id, response FROM results WHERE run_id=? AND benign=0 "
                       "AND blocked=0 AND (judge_llm IS NULL OR judge_llm=-1)", (run,)).fetchall()
    if a.limit:
        rows = rows[: a.limit]
    done = 0
    for rid, aid, resp in rows:
        ok, why = judge.judge(resolve(aid), resp)
        con.execute("UPDATE results SET judge_llm=?, judge_llm_reason=? WHERE rowid=?",
                    (-1 if ok is None else int(ok), why, rid))
        con.commit()
        done += 1
    con.commit()
    print(f"run {run}: judged {done} responses, spent ${tracker.spent:.4f}")
    dis = con.execute("SELECT attack_id, guard, success, judge_llm, substr(response,1,90) FROM results "
                      "WHERE run_id=? AND benign=0 AND blocked=0 AND judge_llm IN (0,1) AND success!=judge_llm", (run,)).fetchall()
    tot = con.execute("SELECT count(*) FROM results WHERE run_id=? AND benign=0 AND blocked=0 AND judge_llm IN (0,1)", (run,)).fetchone()[0]
    print(f"rule judge vs Claude judge: {tot - len(dis)}/{tot} agree; disagreements:")
    for d in dis[:15]:
        print("  ", d)


if __name__ == "__main__":
    main()
