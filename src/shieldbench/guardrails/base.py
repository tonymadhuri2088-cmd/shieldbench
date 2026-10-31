from dataclasses import dataclass


@dataclass
class Verdict:
    blocked: bool
    reason: str = ""
    text: str | None = None  # optional rewritten (e.g. redacted) text
    latency_s: float | None = None  # guard time if the guard measured it (LLM guards)


class Guard:
    name = "g0_none"

    def check_input(self, text: str) -> Verdict:
        return Verdict(False)

    def check_output(self, text: str) -> Verdict:
        return Verdict(False)


BLOCK_MESSAGE = "Sorry, I can't help with that request."
