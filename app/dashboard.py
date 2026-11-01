"""Streamlit dashboard. Reads results/sample_results.csv (or a CSV you export).
Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shieldbench.eval import metrics  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
st.set_page_config(page_title="ShieldBench", layout="wide")
st.title("ShieldBench")
st.caption("Automated red-teaming and guardrail evaluation for LLM apps. Targets are fake apps with fake secrets.")

files = sorted(RES.glob("*.csv"))
if not files:
    st.error("No results CSV found in results/. Run: python -m shieldbench.run --export results/my_run.csv")
    st.stop()
names = [f.name for f in files if not f.name.endswith("_raw.csv") and not f.name.startswith("adaptive") and not f.name.startswith("heldout")] or [f.name for f in files]
if "final_judged.csv" in names:
    names.remove("final_judged.csv"); names.insert(0, "final_judged.csv")   # show the final results first
choice = st.sidebar.selectbox("Results file", names)
df = pd.read_csv(RES / choice)
if choice.startswith("sample"):
    st.warning("SAMPLE DATA from a scripted mock bot. It demonstrates the pipeline only; it says nothing about real LLMs.")
st.sidebar.write(f"Target: **{', '.join(df.target.unique())}**")
st.sidebar.write(f"Rows: {len(df)}")

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Attack success", "Over-refusal and latency", "Mutations", "Examples", "Adaptive attacker", "Judge reliability", "Unseen attacks"])
df = metrics.with_verdict(df)
asr = metrics.asr_table(df, "verdict")
with tab0:
    att = df[df.benign == 0]
    guards = sorted(att.guard.unique())
    st.subheader("Does the guard stop attacks, and what does it cost?")
    cols = st.columns(len(guards))
    ort, lat, cs = metrics.overrefusal_table(df), metrics.latency_table(df), metrics.claude_share(df)
    rows = []
    for col, g in zip(cols, guards):
        grp = att[att.guard == g]
        p, lo, hi = metrics.wilson(int(grp.verdict.sum()), len(grp))
        col.metric(g, f"{p:.1%} attacks succeed")
        col.caption(f"95% interval {lo:.1%} to {hi:.1%} ({len(grp)} attacks)")
        o = ort[ort.guard == g]
        la = lat[lat.guard == g]
        rows.append({"guard": g, "attack success %": round(p * 100, 1), "interval %": f"{lo*100:.1f} to {hi*100:.1f}",
                     "harmless blocked by guard": int(o.guard_blocked.iloc[0]) if len(o) else None,
                     "guard time per request (ms)": round(float(la.guard_median_ms.iloc[0]), 2) if len(la) else None,
                     "Claude called on % of requests": float(cs[cs.guard == g].claude_used_pct.iloc[0]) if (cs.guard == g).any() else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info("Test app: a fake support bot with fake secrets, run on a small local model. Results describe this test set, not every LLM app.")
    with st.expander("Limitations"):
        st.markdown("- Small attack set (about 120 fixed attacks) and 20 adaptive attempts per guard; a 0% result can still hide a true rate of up to about 3% (fixed set) or 16% (adaptive set).\n"
                    "- The adaptive attacker is Claude Haiku and the Claude guard is also Haiku, so the guard may be better at spotting attacks written by its own kind.\n"
                    "- About one in five Claude-guard blocks happened because the classifier refused to read the attack text; these are counted as blocks (fail closed), not as true classifications.\n"
                    "- Harmless prompts are few (60 in total) and fairly easy, so over-refusal is probably underestimated.\n"
                    "- Attack success is decided by an LLM judge (the adaptive run uses a plain canary-string check). Judge agreement with one human labeler on 100 replies is on the Judge reliability tab; it is a single labeler and a small sample.")
with tab1:
    st.subheader("Attack success rate by guard and category (95% Wilson interval)")
    piv = asr.pivot(index="category", columns="guard", values="asr")
    st.bar_chart(piv)
    show = asr.copy()
    for c in ("asr", "ci_low", "ci_high"):
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show)
    st.subheader("How each guard produced its blocks")
    st.dataframe(metrics.block_breakdown(df), hide_index=True)
with tab2:
    st.subheader("Over-refusal on benign prompts")
    st.dataframe(metrics.overrefusal_table(df).round(3))
    st.subheader("Task success on clean documents (does the bot still help?)")
    st.dataframe(metrics.utility_table(df).round(3))
    st.subheader("Latency per guard (seconds, uncached calls)")
    st.dataframe(metrics.latency_table(df).round(4))
with tab3:
    a = df[df.benign == 0]
    st.subheader("Success rate by mutation and guard")
    hm = a.groupby(["mutation", "guard"]).verdict.mean().unstack().round(3)
    st.dataframe(hm)
with tab4:
    st.subheader("Redacted examples")
    cat = st.selectbox("Category", sorted(df[df.benign == 0].category.unique()))
    ex = df[(df.category == cat)].copy()
    ex["response"] = ex["response"].astype(str).str.replace(r"SB-CANARY-\w+|STAFF-ONLY-\w+|[\w.+-]+@[\w-]+\.\w+|\+91-\d+-\d+", "[REDACTED]", regex=True).str[:200]
    st.dataframe(ex[["attack_id", "mutation", "guard", "blocked", "success", "response"]].head(60))

with tab5:
    import json as _json
    adp = sorted(RES.glob("adaptive*.csv"))
    if not adp:
        st.info("No adaptive attacker results yet. Run: python -m shieldbench.attacks.adaptive --export results/adaptive_runs.csv")
    else:
        ep = pd.concat([pd.read_csv(f) for f in adp], ignore_index=True)
        st.subheader("Can an attacker that learns from replies get past each guard?")
        st.caption(f"Attacker: {ep.attacker.iloc[0]} | target: {ep.target.iloc[0]} | {len(ep)} episodes, up to {int(ep.rounds.max())} rounds each")
        at = metrics.adaptive_table(ep)
        st.bar_chart(at.set_index("guard")["adaptive_asr"])
        show = at.copy()
        for c in ("adaptive_asr", "ci_low", "ci_high"):
            show[c] = (show[c] * 100).round(1)
        st.dataframe(show, hide_index=True)
        st.subheader("Example attack conversations (secrets redacted)")
        won = ep[ep.success == 1]
        pick = won if len(won) else ep
        row = pick.iloc[st.number_input("Example number", 0, max(len(pick) - 1, 0), 0)]
        st.write(f"**{row.guard}**, goal `{row.goal}`, {'reached' if row.success else 'not reached'} in {row.rounds} rounds")
        red = r"SB-CANARY-\w+|STAFF-ONLY-\w+|[\w.+-]+@[\w-]+\.\w+|\+91-\d+-\d+"
        for t in _json.loads(row.transcript):
            st.markdown(f"**Round {t['round']} attacker:** {pd.Series([t['message']]).str.replace(red, '[REDACTED]', regex=True).iloc[0][:400]}")
            st.markdown(f"**Bot:** {pd.Series([t['reply']]).str.replace(red, '[REDACTED]', regex=True).iloc[0][:300]}  _(blocked by: {t['blocked_by'] or 'nothing'})_")

with tab6:
    LAB = Path(__file__).resolve().parents[1] / "data" / "labels"
    hp, kp = LAB / "human_labels3.csv", LAB / "key.csv"
    st.subheader("Is the judge right? Checked against hand labels")
    if not (hp.exists() and kp.exists()):
        st.info("Hand-label files not found in data/labels/ (human_labels3.csv and key.csv).")
    else:
        hum = pd.read_csv(hp)[["item_id", "label"]].dropna()
        key = pd.read_csv(kp).merge(hum, on="item_id")
        key["label"] = key.label.astype(int)

        def _kappa(a, b):
            n = len(a)
            po = sum(x == y for x, y in zip(a, b)) / n
            pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in set(a) | set(b))
            return 1.0 if pe == 1 else (po - pe) / (1 - pe)

        rows = []
        for name, label in (("rule_judge", "Rule judge (canary string match)"), ("claude_judge", "Claude judge (Sonnet, per-category rubric)")):
            k = key[key[name].isin([0, 1])]
            h, j = k.label.tolist(), k[name].astype(int).tolist()
            agree = sum(x == y for x, y in zip(h, j))
            lo, hi = metrics.wilson(agree, len(k))[1:]
            rows.append({"judge": label, "replies scored": len(k), "agrees with human": agree,
                         "agreement %": round(100 * agree / len(k), 1), "interval %": f"{lo*100:.0f} to {hi*100:.0f}",
                         "kappa": round(_kappa(h, j), 2),
                         "false alarms": sum(x == 0 and y == 1 for x, y in zip(h, j)),
                         "misses": sum(x == 1 and y == 0 for x, y in zip(h, j))})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"{len(key)} replies were labeled by hand (1 labeler); {int(key.label.sum())} were attacks that worked. "
                   "False alarm = judge said the attack worked but the human said no. Miss = the reverse.")
        st.markdown("**How to read this honestly**")
        st.markdown("- The first, fully blind pass agreed with the Claude judge on 89% of replies (kappa 0.53). Most disagreements were hasty human errors.\n"
                    "- The human then re-read the 17 replies where they disagreed with a judge, after seeing that they disagreed. That makes the figures above optimistic.\n"
                    "- The rule judge counts a refusal that quotes the secret or the attack word as a leak. This is why the Claude judge decides the final verdict.\n"
                    "- A few replies could not be scored by the Claude judge and are left out of its row.\n"
                    "- One labeler and 100 replies is a small check, not proof.")


with tab7:
    st.subheader("Attacks written after the detector was trained")
    parts = [RES / "heldout_judged.csv", RES / "heldout_v2_judged.csv"]
    if not all(p.exists() for p in parts):
        st.info("Unseen-attack results not found (results/heldout_judged.csv and results/heldout_v2_judged.csv).")
    else:
        old = metrics.with_verdict(pd.read_csv(parts[0]))
        new = metrics.with_verdict(pd.read_csv(parts[1]))
        old, new_df = old[old.guard.isin(["g0_none", "g1_regex", "g3_claude"])], new
        hd = pd.concat([old, new_df], ignore_index=True)
        rows = []
        for g in sorted(hd.guard.unique()):
            a = hd[(hd.guard == g) & (hd.benign == 0)]
            h = hd[(hd.guard == g) & (hd.benign == 1)]
            p, lo, hi = metrics.wilson(int(a.verdict.sum()), len(a))
            rows.append({"guard": g, "attacks that worked": f"{int(a.verdict.sum())} of {len(a)}", "attack success %": round(p * 100, 1),
                         "interval %": f"{lo*100:.1f} to {hi*100:.1f}", "harmless blocked": f"{int(h.blocked.sum())} of {len(h)}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown("40 new attacks (long cover stories, JSON and code wrappers, spaced or reversed text, Tamil-English, fake transcripts, "
                    "documents shaped like reviews or invoices) and 20 new harmless prompts. Their closest match in the training data is below 0.7 "
                    "(similarity, 1 means identical).")
        st.warning("Before retraining, the detector (G2) let 2 of these 40 through and the cascade (G4) let 1 through. "
                   "Training examples in those two styles were then added, which fixed them. Because the styles were chosen after "
                   "looking at this set, it is now a tuning set, not an untouched test. A fresh set is needed to claim more.")
