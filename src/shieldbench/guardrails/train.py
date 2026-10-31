"""Train and honestly evaluate the G2 detector.
   python -m shieldbench.guardrails.train
Prints: 5-fold cross-validation, and leave-one-category-out recall (the detector is trained WITHOUT one
attack category and then asked to catch it, which tests whether it generalises beyond what it has seen).
Note: cross-validation on generated text is optimistic (similar sentences land on both sides of a split);
the real test is the fixed attack set and the adaptive attacker, which the detector never saw."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config
from .detector_guard import MODEL_PATH, expand, fit_model, build_pipeline


def _pick_thresholds(y, p):
    """hi: lowest score that keeps harmless false positives under 1%. lo: highest score that misses under 1% of attacks."""
    ben, atk = np.sort(p[y == 0]), np.sort(p[y == 1])
    hi = float(np.quantile(ben, 0.99)) if len(ben) else 0.9
    lo = float(np.quantile(atk, 0.01)) if len(atk) else 0.1
    return float(min(max(hi, 0.6), 0.98)), float(min(max(lo, 0.02), 0.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.ROOT / "data" / "train" / "train.csv"))
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--C", type=float, default=4.0)
    a = ap.parse_args()
    import joblib
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    df = pd.read_csv(a.data).dropna(subset=["text"]).reset_index(drop=True)
    X, y = np.array([expand(t) for t in df.text], dtype=object), df.label.to_numpy()
    print(f"training rows: {len(df)} ({int(y.sum())} attacks, {int((1 - y).sum())} harmless)")

    skf = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(build_pipeline(a.C), X, y, cv=skf, method="predict_proba")[:, 1]
    pred = oof >= 0.5
    print(f"\n5-fold cross-validation at threshold 0.5:  attack recall {pred[y == 1].mean():.1%}   "
          f"harmless wrongly blocked {pred[y == 0].mean():.1%}")

    print("\nLeave-one-category-out (trained WITHOUT the category, then asked to catch it):")
    rows = []
    for cat in sorted(df[df.label == 1].category.unique()):
        te = (df.category == cat).to_numpy()
        m = build_pipeline(a.C).fit(X[~te], y[~te])
        p = m.predict_proba(X[te])[:, 1]
        rows.append({"held-out category": cat, "n": int(te.sum()), "attacks caught at 0.5": round(float((p >= 0.5).mean()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    hi, lo = _pick_thresholds(y, oof)
    model = fit_model(df.text, y, a.C)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thr": 0.5, "hi": hi, "lo": lo,
                 "meta": {"rows": int(len(df)), "attacks": int(y.sum()), "C": a.C, "seed": config.SEED}}, a.out)
    print(f"\nsaved {a.out}   block threshold 0.5   cascade: block >= {hi:.2f}, allow <= {lo:.2f}, ask Claude in between")


if __name__ == "__main__":
    main()
