"""Honest scorecard for the G2 detector on prompts it was NOT trained on.
   python -m shieldbench.guardrails.detector_eval
For each test set it prints precision, recall, F1, false-positive rate at the block threshold, AUC and ECE
(expected calibration error: how far the score is from the real attack rate, lower is better, 0 is perfect).
Sets: the main test set (124 attacks, 60 harmless), and the two held-out sets (40 attacks, 20 harmless each).
Also prints, for the cascade thresholds, how many prompts fall between 'allow' and 'block' (those go to Claude)."""
import numpy as np
import pandas as pd
from ..attacks import mutators
from ..attacks.heldout import HELDOUT, HELDOUT_BENIGN
from ..attacks.heldout2 import HELDOUT2, HELDOUT2_BENIGN
from ..attacks.heldout3 import HELDOUT3, HELDOUT3_BENIGN
from ..attacks.library import ATTACKS, BENIGN, BENIGN_HARD, controls
from .detector_guard import DetectorGuard

MUTS = ["base64", "leetspeak", "roleplay", "hinglish", "multi_turn"]


def _text(a):
    return a.text + (" " + a.document if a.document else "")


def test_sets():
    main_atk = ATTACKS + mutators.apply(ATTACKS, MUTS)
    main_ben = list(BENIGN) + list(BENIGN_HARD) + [_text(c) for c in controls()]
    return {
        "main test set": ([_text(a) for a in main_atk], main_ben),
        "unseen set 1 (tuned on)": ([_text(a) for a in HELDOUT], list(HELDOUT_BENIGN)),
        "unseen set 2 (untouched)": ([_text(a) for a in HELDOUT2], list(HELDOUT2_BENIGN)),
        "jailbreak set 3 (untouched)": ([_text(a) for a in HELDOUT3], list(HELDOUT3_BENIGN)),
    }


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def scorecard(guard: DetectorGuard, sets=None) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    lo, hi = guard.bundle.get("lo", 0.1), guard.bundle.get("hi", 0.9)
    for name, (atk, ben) in (sets or test_sets()).items():
        texts = list(atk) + list(ben)
        y = np.array([1] * len(atk) + [0] * len(ben))
        p = np.array([guard.proba(t) for t in texts])
        pred = p >= guard.threshold
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
        unsure = float(((p > lo) & (p < hi)).mean())
        rows.append({"set": name, "attacks": len(atk), "harmless": len(ben), "precision": round(prec, 3), "recall": round(rec, 3),
                     "F1": round(f1, 3), "false positive rate": round(fp / (fp + tn), 3), "AUC": round(float(roc_auc_score(y, p)), 3),
                     "ECE": round(ece(y, p), 3), "sent to Claude by cascade": f"{unsure:.0%}"})
    return pd.DataFrame(rows)


def main():
    g = DetectorGuard()
    pd.set_option("display.width", 220)
    print(f"G2 detector at block threshold {g.threshold}; cascade allows <= {g.bundle.get('lo')} and blocks >= {g.bundle.get('hi')}\n")
    print(scorecard(g).to_string(index=False))
    print("\nprecision = of the prompts blocked, how many were attacks; recall = of the attacks, how many were blocked;"
          "\nECE close to 0 means the score can be read as a probability.")


if __name__ == "__main__":
    main()
