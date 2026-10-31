from .base import Guard
from .regex_guard import RegexGuard

GUARDS = {"g0": Guard, "g1": RegexGuard}  # g2 detector and g3 Claude guard plug in here later


def make_guard(key: str, store=None, tracker=None, client=None) -> Guard:
    if key == "g3":
        from .claude_guard import ClaudeGuard
        return ClaudeGuard(store, tracker=tracker, client=client)
    if key not in GUARDS:
        raise KeyError(f"unknown guard {key!r}; available: {sorted(GUARDS) + ['g3']}")
    return GUARDS[key]()
