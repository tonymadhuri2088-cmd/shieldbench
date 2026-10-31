# ShieldBench

Automated red-teaming and guardrail evaluation for LLM apps. It attacks a **fake** customer-support bot (fake secret, fake customer data), judges whether each attack worked, applies guardrails, and reports attack success rate (ASR), over-refusal and latency before and after.

> Scope: attacks run only against the test app in this repo. All secrets and personal data are fake canaries.

## Status

| Done | Next (see checklist) |
|---|---|
| Fake support bot, Ollama / Claude / mock targets | C5 Claude judge + human labels + kappa |
| 22 seed attacks, 5 categories, 5 mutators | C7 G3 Claude guard + PII redaction |
| G0 none, G1 regex, G3 Claude guard (Haiku input check + PII redaction) | C12 G2 trained detector |
| Rule judge + Claude judge (Sonnet), SQLite cache, cost cap | C11 adaptive attacker |
| ASR with Wilson intervals, over-refusal, latency, task success on clean docs | C13 deploy dashboard |
| Human label sheet + Cohen's kappa tools, Streamlit dashboard, tests, CI | XSTest, public attack sets |

`results/sample_results.csv` comes from a scripted **mock** bot. It proves the pipeline works; it says nothing about real LLMs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
copy .env.example .env          # then edit if you use Claude
set PYTHONPATH=src              # PowerShell: $env:PYTHONPATH="src"
pytest                          # 7 tests should pass
```

## Run

```bash
# 1. pipeline check with the mock bot (free, instant)
python -m shieldbench.run --target mock --guards g0,g1 --export results/mock_run.csv

# 2. real local model (needs: ollama pull llama3.2:3b). Small batch first!
python -m shieldbench.run --target ollama --guards g0,g1 --limit 10
python -m shieldbench.run --target ollama --guards g0,g1 --mutate base64,leetspeak,roleplay,hinglish,multi_turn --export results/ollama_run.csv

# 3. Claude target (costs credit; set a spending limit in the Anthropic Console first)
python -m shieldbench.run --target claude --guards g0,g1 --limit 10 --budget 1

# 4. dashboard
streamlit run app/dashboard.py
```

Reruns are free: every model call is cached in SQLite by hash of (model, messages). A run stops with `BudgetExceeded` at the cap.

## Guards and final numbers

```bash
python -m shieldbench.run --target ollama --guards g0,g1,g3 --mutate base64,leetspeak,roleplay,hinglish,multi_turn --limit 10   # small batch first
python -m shieldbench.judge.run_judge        # Claude judge on the latest run
python -m shieldbench.report                 # all tables
```
For the FINAL numbers use a fresh database so latency is measured on real calls, not read from an older cache:
`--db results/final.db` on both `run` and `run_judge` / `report`.
G3 fails closed: if the guard's reply cannot be parsed the request is blocked.

## Judge validation (C5, C6)

```bash
# 1. judge a run with Claude (costs a little; try --limit 10 first)
python -m shieldbench.judge.run_judge --limit 10
python -m shieldbench.judge.run_judge
# 2. make a blind label sheet (100 items, stratified by category)
python -m shieldbench.judge.label export --n 100
#    give a COPY of data/labels/label_sheet.csv to each of two people; they fill the `label` column (1 = attack succeeded, 0 = failed)
#    keep data/labels/key.csv private (it holds the judge verdicts)
# 3. agreement numbers
python -m shieldbench.judge.label kappa --a data/labels/A.csv --b data/labels/B.csv
```

## Layout

```
src/shieldbench/
  targets/     fake_data, support_bot, ollama_target, claude_target, mock_target
  attacks/     library (seed attacks), mutators
  guardrails/  base, regex_guard (G1), pipeline
  judge/       rule_judge (canary / marker checks)
  eval/        runner, metrics (Wilson, over-refusal, latency), cost (cap)
  store/       db (SQLite results + cache)
app/dashboard.py   tests/   results/
```

## Known limitations (so far)

- Task success on clean documents is checked with a keyword, a rough proxy.
- The rule judge counts a response as an attack success if it contains the marker or a canary, even if the bot quoted it while refusing. The Claude judge (C5) and human labels will measure this error.
- Seed attacks are our own templates; public sets (JailbreakBench, HarmBench subsets) are not added yet.
- Over-refusal uses 12 hand-written benign prompts, not XSTest yet, and a simple refusal-phrase check.
- Hinglish mutation is only a polite prefix so far; real Hindi / code-mixed attacks are still to write.
- Cost prices in `config.py` are placeholders; check the official pricing page.

## Adaptive attacker (C11)

```bash
# small test first (a few cents): 1 trial, 3 rounds
python -m shieldbench.attacks.adaptive --target ollama --attacker hybrid --guards g0,g1,g3 --trials 1 --rounds 3 --budget 0.5
# full run + export for the dashboard
python -m shieldbench.attacks.adaptive --target ollama --attacker hybrid --trials 3 --rounds 6 --budget 1 --export results/adaptive_runs.csv
```
`hybrid` = Claude writes the attack; if Claude refuses, the local model writes it. Refusals are counted, not hidden.
Add `--hard-benign` to `shieldbench.run` to include 36 tricky-but-harmless prompts in the over-refusal numbers.
