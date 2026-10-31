"""G2: a small trained detector (TF-IDF character n-grams + logistic regression).
Runs locally in about a millisecond and costs nothing per request. It is trained on a SEPARATE
set of examples (data/train/train.csv), never on the test attacks, so its scores on the test set are honest.
Before scoring, the text is expanded with decoded base64 pieces and a de-leetspeak copy, so simple
disguises do not hide the attack words."""
import base64
import re
import time
from pathlib import Path
from .. import config
from .base import Guard, Verdict
from .claude_guard import redact

MODEL_PATH = config.ROOT / "data" / "models" / "g2.joblib"
_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_UNLEET = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})


def _decode_b64(text: str) -> list[str]:
    out = []
    for tok in _B64.findall(text):
        try:
            raw = base64.b64decode(tok + "=" * (-len(tok) % 4), validate=True)
            s = raw.decode("utf-8")
        except Exception:
            continue
        if s and sum(c.isprintable() for c in s) / len(s) > 0.9:
            out.append(s)
    return out


def expand(text: str) -> str:
    """The original text, plus decoded base64 pieces, plus a de-leetspeak copy."""
    parts = [text] + _decode_b64(text)
    parts.append(text.translate(_UNLEET))
    return " \n ".join(parts)


def build_pipeline(C: float = 4.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("tfidf", feats), ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def fit_model(texts, labels, C: float = 4.0):
    pipe = build_pipeline(C)
    pipe.fit([expand(t) for t in texts], list(labels))
    return pipe


def load_bundle(path: Path | str = MODEL_PATH):
    import joblib
    return joblib.load(path)


class DetectorGuard(Guard):
    name = "g2_detector"

    def __init__(self, bundle: dict | None = None, threshold: float | None = None):
        self.bundle = bundle or load_bundle()
        self.model = self.bundle["model"]
        self.threshold = threshold if threshold is not None else self.bundle.get("thr", 0.5)

    def proba(self, text: str) -> float:
        return float(self.model.predict_proba([expand(text)])[0][1])

    def check_input(self, text):
        t0 = time.perf_counter()
        p = self.proba(text)
        lat = time.perf_counter() - t0
        return Verdict(p >= self.threshold, f"detector score {p:.2f}" if p >= self.threshold else "", latency_s=lat)

    def check_output(self, text):
        red = redact(text)
        if red != text:
            return Verdict(False, "pii redacted", text=red, latency_s=0.0)
        return Verdict(False, latency_s=0.0)
