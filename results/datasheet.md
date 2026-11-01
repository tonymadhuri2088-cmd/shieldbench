# ShieldBench datasheet (code version 0.33)

A fake support bot (AcmeBot) with fake secrets. Nothing here touches a real system.

## What was tested
| set | attacks | harmless | content hash |
|---|---|---|---|
| main (5 disguises added at run time) | 22 seeds (124 with disguises) | 60 | c4f67ee72d947614 |
| unseen set 1 (tuning set) | 40 | 20 | e25f1d813ade746b |
| unseen set 2 (untouched, run once) | 40 | 20 | 518f0daa9958721b |
| jailbreak set 3 (untouched, run once) | 40 | 20 | 29711533c2088d88 |
| advanced jailbreak set 4 (untouched, run once) | 40 | 20 | 23b1543ac149cabc |

## Detector (G2)
- training file hash: a2df4dbfc85d6528
- model file hash: 5031ad5f74eea965
- features: character 2-5 grams and word 1-2 grams; logistic regression, class weights balanced

## Results file
- results/final_judged.csv: 920 rows, target ollama:llama3.2:3b
- guards: g0_none, g1_regex, g2_detector, g3_claude, g4_cascade
- attack rows: 620, harmless rows: 300

## Known limits
- One small local target model; one human labeler; attacker and Claude guard are the same model family.
- Unseen set 1 was used to improve the detector, so it is not a clean test. Set 2 was not.
- A result of 0 means the true rate is probably under a few percent, not exactly zero.
