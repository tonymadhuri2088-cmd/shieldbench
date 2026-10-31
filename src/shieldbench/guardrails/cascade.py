"""G4: a cascade. Cheap checks first, the Claude guard only when the cheap checks are unsure.
  1. regex rules block obvious attacks (microseconds)
  2. the trained detector gives a score: >= hi blocks, <= lo allows
  3. only scores between lo and hi are sent to the Claude guard (about 1 s, costs money)
The aim: keep most of the Claude guard's protection while paying its delay and cost only on hard cases."""
import time
from .base import Guard, Verdict
from .claude_guard import redact
from .regex_guard import RegexGuard


class CascadeGuard(Guard):
    name = "g4_cascade"

    def __init__(self, detector, claude_guard, lo: float | None = None, hi: float | None = None):
        self.regex, self.detector, self.claude = RegexGuard(), detector, claude_guard
        b = getattr(detector, "bundle", {}) or {}
        self.lo = lo if lo is not None else b.get("lo", 0.1)
        self.hi = hi if hi is not None else b.get("hi", 0.9)
        self.stats = {"regex": 0, "detector_block": 0, "detector_allow": 0, "claude": 0}

    def check_input(self, text):
        t0 = time.perf_counter()
        v = self.regex.check_input(text)
        if v.blocked:
            self.stats["regex"] += 1
            return Verdict(True, "cascade: " + v.reason, latency_s=time.perf_counter() - t0)
        p = self.detector.proba(text)
        local = time.perf_counter() - t0
        if p >= self.hi:
            self.stats["detector_block"] += 1
            return Verdict(True, f"cascade: detector score {p:.2f}", latency_s=local)
        if p <= self.lo:
            self.stats["detector_allow"] += 1
            return Verdict(False, "", latency_s=local)
        self.stats["claude"] += 1
        c = self.claude.check_input(text)
        reason = ("cascade: Claude guard: " + c.reason) if c.blocked else ""
        return Verdict(c.blocked, reason, latency_s=local + (c.latency_s or 0.0))

    def check_output(self, text):
        red = redact(text)
        if red != text:
            return Verdict(False, "pii redacted", text=red, latency_s=0.0)
        return Verdict(False, latency_s=0.0)
