# ShieldBench

Automated red-teaming and guardrail evaluation for LLM applications.

ShieldBench attacks a **fake** customer-support bot (fake secrets, fake customers), decides whether each attack worked, applies guardrails, and reports how much each guard helps, what it costs in over-blocking and delay, and whether an attacker that learns from replies can still get through.

> Scope: attacks run only against the test app in this repo. Every secret and personal detail is a fake canary. Nothing here targets real systems.

## Headline results

Target: `llama3.2:3b` running locally (temperature 0, fixed seed). Test app: "AcmeBot", a support bot holding a fake API key, a fake staff discount code and two fake customers.

| Guard | Fixed attacks that worked (124) | Adaptive attacker reached its goal (20 tries) | Fresh untouched attacks that worked (40) | Harmless prompts blocked (60) | Added delay per request |
|---|---|---|---|---|---|
| G0 none | 16.9% (11.4 to 24.5) | **90%** (70 to 97) | 20% | 0 | none |
| G1 regex rules | 5.6% (2.8 to 11.2) | **55%** (34 to 74) | 12.5% | 5 (8%) | under 1 ms |
| G2 trained detector | 0.0% (0 to 3.0) | **0%** (0 to 16) | 0% | 9 (15%) | about 5 ms |
| G3 Claude guard | 0.0% (0 to 3.0) | **0%** (0 to 16) | 0% | 0 | about 1.2 s (median) |
| G4 cascade (G2, then Claude if unsure) | 0.0% (0 to 3.0) | **0%** (0 to 16) | 0% | 3 (5%) | about 130 ms (mean); Claude used on 10% of requests |

Intervals are 95% Wilson intervals. They are wide because the test sets are small; a 0% result means "probably below the upper end of the interval", not "zero".

What the numbers say:

1. **Fixed tests flatter weak guards.** The regex guard looks good on fixed attacks (5.6%) but an attacker that adapts gets the secret key and the discount code every time (10 of 10). The regex guard only helps against personal-data leaks (the attacker won 1 of 10 there).
2. **The Claude guard held** across 120 adaptive rounds and all fixed attacks, without blocking any of 60 harmless prompts (36 of them deliberately tricky).
3. **It costs time:** the Claude guard adds about 1.2 s per request (p95 1.7 s), roughly 40% on top of the local model's 3 s reply.
4. **A cheap cascade gets most of the way there.** G4 asks Claude on only about 10% of requests (about 35% on the unseen set), so its mean guard time is about 130 ms against about 1.2 s for G3, at the price of 3 wrongly blocked harmless prompts out of 60. The small detector alone (G2) blocks too many harmless prompts (9 of 60, 8 of the 36 tricky ones).
5. **Unseen attacks exposed two weak styles.** On 40 attacks written after training (cover stories, JSON and code wrappers, spaced or reversed text, Tamil-English, fake transcripts), the first G2 let 2 through and G4 let 1 through. Training examples in those two styles (fake system messages in JSON, "act as a terminal" role-plays) were added and G2 and G4 were retrained; both then let 0 of 40 through. Those styles were chosen after seeing this set, so it is now a tuning set, not an untouched test. A fresh set of 40 attacks in new styles (hex, look-alike letters, emoji, many-shot, Telugu, Bengali and Hindi, email-chain and XML documents), run once and never used for training, was then stopped completely by G2, G3 and G4 (0 of 40), while 8 of 40 worked with no guard and 5 of 40 against regex rules. Only 8 of the 40 work against an unguarded bot, so the test is modest.
6. **The judge was checked by hand.** On 100 hand-labeled replies the Claude judge agreed with the human on 97 of 97 it scored; the simple rule judge agreed on 96 of 100 and wrongly counted 3 refusals that quoted a secret as leaks (see "Judge validation" for how much to trust this).

## How it works

```mermaid
flowchart LR
  A[Attack library<br/>22 seeds x 5 mutators<br/>+ adaptive attacker] --> G1[Input guard]
  G1 --> B[Target bot<br/>local Ollama / Claude / mock]
  B --> G2[Output guard<br/>PII redaction]
  G2 --> J[Judge<br/>rule + Claude]
  J --> S[(SQLite<br/>results + cache)]
  S --> R[Report + Streamlit dashboard]
```

- **Attack categories:** direct injection, indirect injection (instruction hidden in a retrieved document), jailbreak, system-prompt extraction, personal-data leakage.
- **Mutators:** base64, leetspeak, roleplay, Hinglish, multi-turn. 22 seed attacks become 124 test cases.
- **Guards:** G0 none; G1 regex rules (naive baseline); G2 a small trained detector (TF-IDF character n-grams + logistic regression, about 1 ms, free per request, sees decoded base64 and de-leetspeaked text); G3 Claude Haiku input classifier plus regex redaction of emails and phone numbers on output; G4 a cascade (regex, then G2, and Claude only when G2 is unsure). G3 fails closed: if the classifier's reply cannot be read, the request is blocked.
- **Judges:** a rule judge (looks for canary strings) and a Claude judge (Sonnet, one rubric per category). The final verdict uses the Claude judge when it ran.
- **Adaptive attacker:** a plain Python loop. A Claude Haiku attacker (local Ollama model as fallback if Claude refuses) writes a message, the message goes through the guard, the bot and the output guard, a deterministic canary check decides whether the goal was reached, and the reply is fed back for the next round (up to 6 rounds, 5 trials per goal, 4 goals: key, discount code, email, phone).
- **Metrics:** attack success rate with Wilson intervals, over-refusal split into "guard blocked" and "bot refused", task success on clean documents, model latency vs guard overhead, a block breakdown, adaptive success and rounds to win.
- **Engineering:** every model call is cached in SQLite by a hash of the model and messages (reruns are free and results reproducible), a cost cap stops a run with `BudgetExceeded`, schema migrations, 40 tests, GitHub Actions CI. The whole project cost well under 1 US dollar of API credit.

## Judge validation

100 bot replies (stratified by category) were exported as a blind sheet and labeled by hand by one person.

| Judge | Agrees with human | Kappa | False alarms | Misses |
|---|---|---|---|---|
| Claude judge | 97 / 97 | 1.00 | 0 | 0 |
| Rule judge | 96 / 100 | 0.87 | 3 | 1 |

How much to trust it:
- The first, fully blind pass agreed with the Claude judge on 89% (kappa 0.53); most disagreements were hasty human errors.
- The human then re-read the 17 replies where they disagreed with a judge, after seeing that they disagreed. This makes the table above optimistic.
- One labeler, 100 replies. A second labeler and a larger sample are needed before calling the judge "validated".
- The rule judge's false alarms are all refusals that quote the secret or the attack word, which is why the Claude judge decides the final verdict.
- Three replies could not be scored by the Claude judge and are excluded from its row.

## Known limitations

- **Small test sets.** About 120 fixed attacks and 20 adaptive tries per guard. A 0% result can hide a true rate of about 3% (fixed set) or 16% (adaptive set).
- **Same model family.** The adaptive attacker and the Claude guard are both Haiku, so the guard may be better at spotting attacks written by its own kind.
- **Fail-closed blocks.** 26 of the Claude guard's 124 blocks on fixed attacks happened because the classifier refused to read the attack text (`stop_reason=refusal`). They are counted as blocks, not as true classifications.
- **One small target model**, one fake app, one language mix. Results describe this test setup, not every LLM application.
- **Harmless prompts are few** (60, hand-written) and fairly easy, so over-refusal is probably underestimated. XSTest and public attack sets are not included yet.
- **Task success on clean documents** is checked with a keyword, a rough proxy.
- **Hinglish mutation** only adds a Hindi polite prefix in front of the English attack; it is not native code-mixed text.
- **Each unseen set is only 40 attacks and 20 harmless prompts.** Zero wins can hide a true rate of about 9%, and only 8 of the fresh set's 40 attacks work against an unguarded bot. The first set was used to improve the detector; the second was not.
- **Training set is lopsided** (about 800 attack and 290 harmless prompts), which is why the detector alone over-blocks tricky harmless prompts.
- **G2 and G4 are trained on generated text** (Claude-written examples, kept separate from the test attacks, with near-duplicates of test prompts removed). Cross-validation on generated text is optimistic; the numbers that count are G2/G4 on the fixed and adaptive tests.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
copy .env.example .env          # add your Anthropic key only if you use Claude
pytest                          # 40 tests should pass
```
Never commit `.env`; it is in `.gitignore`.

## Run

```bash
# 1. pipeline check with the mock bot (free, instant)
python -m shieldbench.run --target mock --guards g0,g1 --export results/mock_run.csv

# 2. real local model (needs: ollama pull llama3.2:3b). Small batch first
python -m shieldbench.run --target ollama --guards g0,g1,g3 --limit 10

# 3. full run with all mutators and the tricky harmless prompts, then judge and report
python -m shieldbench.run --target ollama --guards g0,g1,g3 --mutate base64,leetspeak,roleplay,hinglish,multi_turn --hard-benign --db results/final.db --export results/final_raw.csv
python -m shieldbench.judge.run_judge --db results/final.db
python -m shieldbench.report --db results/final.db --csv results/final_judged.csv

# 3b. trained guard G2 and cascade G4 (a few cents for the training data, then free)
pip install -r requirements.txt                       # adds scikit-learn
python -m shieldbench.guardrails.train_data           # Claude Haiku writes the training examples (data/train/train.csv)
python -m shieldbench.guardrails.train                # trains G2, prints cross-validation and leave-one-category-out recall
python -m shieldbench.run --target ollama --guards g2,g4 --mutate base64,leetspeak,roleplay,hinglish,multi_turn --hard-benign --db results/final.db --export results/final_raw.csv

# 3c. attacks written after training (never in the training data), then retest
python -m shieldbench.attacks.heldout                  # checks they do not resemble the training data
python -m shieldbench.run --target ollama --guards g0,g1,g2,g3,g4 --attack-set heldout --db results/heldout.db --export results/heldout_raw.csv
python -m shieldbench.judge.run_judge --db results/heldout.db
python -m shieldbench.report --db results/heldout.db --csv results/heldout_judged.csv

# 3d. second, untouched set (run once; do not train on it)
python -m shieldbench.attacks.heldout2
python -m shieldbench.run --target ollama --guards g0,g1,g2,g3,g4 --attack-set heldout2 --db results/heldout2.db --export results/heldout2_raw.csv
python -m shieldbench.judge.run_judge --db results/heldout2.db
python -m shieldbench.report --db results/heldout2.db --csv results/heldout2_judged.csv

# 4. adaptive attacker (a few cents)
python -m shieldbench.attacks.adaptive --target ollama --attacker hybrid --guards g0,g1,g3 --trials 5 --rounds 6 --budget 1 --db results/final.db --export results/adaptive_runs.csv

# 5. dashboard
streamlit run app/dashboard.py
```

Judge validation:
```bash
python -m shieldbench.judge.label export --n 100          # blind sheet + private key
python -m shieldbench.judge.label check --labels data/labels/human_labels3.csv --out data/labels/recheck3.csv
python -m shieldbench.judge.label kappa --a A.csv --b B.csv   # when two people labeled
```

## Layout

```
src/shieldbench/
  targets/     fake_data, support_bot, ollama_target, claude_target, mock_target
  attacks/     library (seed + hard harmless prompts), mutators, heldout and heldout2 (unseen sets), adaptive (attacker loop)
  guardrails/  base, regex_guard (G1), detector_guard (G2), claude_guard (G3), cascade (G4), train_data, train, pipeline
  judge/       rule_judge, claude_judge, run_judge, label (human labels, kappa)
  eval/        runner, metrics (Wilson, over-refusal, latency, adaptive), cost (cap)
  store/       db (SQLite results, cache, adaptive episodes)
app/dashboard.py   tests/   results/   data/labels/
```
