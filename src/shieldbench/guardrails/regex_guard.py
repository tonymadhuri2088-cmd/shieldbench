"""G1: simple keyword/regex baseline. Deliberately naive: it is the baseline the
better guards must beat."""
import re
from .base import Guard, Verdict

INPUT_PATTERNS = [
    r"ignore (all |your |any )?(the )?(previous|prior|above)? ?(instructions|rules)",
    r"disregard (your|all|the) (rules|instructions)",
    r"(reveal|print|repeat|show).{0,30}(system prompt|instructions|api key)",
    r"developer mode",
    r"no restrictions",
    r"system override",
]
OUTPUT_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.]+",             # email
    r"\+?\d{2}[- ]?\d{5}[- ]?\d{5}",        # phone (Indian style)
    r"\b(sk|api|key)[-_][A-Za-z0-9-]{6,}",  # generic key-looking token
]


class RegexGuard(Guard):
    name = "g1_regex"

    def __init__(self):
        self.inp = [re.compile(p, re.I) for p in INPUT_PATTERNS]
        self.out = [re.compile(p, re.I) for p in OUTPUT_PATTERNS]

    def check_input(self, text):
        for p in self.inp:
            if p.search(text):
                return Verdict(True, f"input matched /{p.pattern}/")
        return Verdict(False)

    def check_output(self, text):
        for p in self.out:
            if p.search(text):
                return Verdict(True, f"output matched /{p.pattern}/")
        return Verdict(False)
