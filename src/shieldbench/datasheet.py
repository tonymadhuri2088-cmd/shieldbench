"""Datasheet: one page that says exactly what was tested, so a result can be reproduced and checked.
   python -m shieldbench.datasheet --csv results/final_judged.csv --out results/datasheet.md
Includes a config hash: the SHA-256 of every attack text, every harmless prompt and the training file, so anyone can
confirm two runs used the same tests."""
import argparse
import hashlib
from pathlib import Path
from . import __version__, config


def _h(parts) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8", "ignore") + b"\x00")
    return m.hexdigest()[:16]


def build(csv: str | None = None) -> str:
    from .attacks.heldout import HELDOUT, HELDOUT_BENIGN
    from .attacks.heldout2 import HELDOUT2, HELDOUT2_BENIGN
    from .attacks.library import ATTACKS, BENIGN, BENIGN_HARD, controls
    t = lambda a: a.text + (a.document or "")
    tr = config.ROOT / "data" / "train" / "train.csv"
    mdl = config.ROOT / "data" / "models" / "g2.joblib"
    fh = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "not present"
    L = [f"# ShieldBench datasheet (code version {__version__})", "",
         "A fake support bot (AcmeBot) with fake secrets. Nothing here touches a real system.", "",
         "## What was tested", "| set | attacks | harmless | content hash |", "|---|---|---|---|",
         f"| main (5 disguises added at run time) | {len(ATTACKS)} seeds (124 with disguises) | {len(BENIGN) + len(BENIGN_HARD) + len(controls())} | {_h([t(a) for a in ATTACKS] + list(BENIGN) + list(BENIGN_HARD) + [t(c) for c in controls()])} |",
         f"| unseen set 1 (tuning set) | {len(HELDOUT)} | {len(HELDOUT_BENIGN)} | {_h([t(a) for a in HELDOUT] + list(HELDOUT_BENIGN))} |",
         f"| unseen set 2 (untouched, run once) | {len(HELDOUT2)} | {len(HELDOUT2_BENIGN)} | {_h([t(a) for a in HELDOUT2] + list(HELDOUT2_BENIGN))} |", "",
         "## Detector (G2)", f"- training file hash: {fh(tr)}", f"- model file hash: {fh(mdl)}", "- features: character 2-5 grams and word 1-2 grams; logistic regression, class weights balanced", ""]
    if csv and Path(csv).exists():
        import pandas as pd
        from .eval import metrics
        d = metrics.with_verdict(pd.read_csv(csv))
        L += ["## Results file", f"- {csv}: {len(d)} rows, target {', '.join(map(str, d.target.unique()))}",
              f"- guards: {', '.join(sorted(d.guard.unique()))}", f"- attack rows: {int((d.benign == 0).sum())}, harmless rows: {int((d.benign == 1).sum())}", ""]
    L += ["## Known limits", "- One small local target model; one human labeler; attacker and Claude guard are the same model family.",
          "- Unseen set 1 was used to improve the detector, so it is not a clean test. Set 2 was not.",
          "- A result of 0 means the true rate is probably under a few percent, not exactly zero."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    s = build(a.csv or None)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
        print("wrote", a.out)
    print(s)


if __name__ == "__main__":
    main()
