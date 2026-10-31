from .base import Guard
from .regex_guard import RegexGuard

GUARDS = {"g0": Guard, "g1": RegexGuard}  # g2 (trained detector), g3 (Claude guard), g4 (cascade) are built in make_guard


def make_guard(key: str, store=None, tracker=None, client=None) -> Guard:
    if key == "g2":
        from .detector_guard import DetectorGuard
        return DetectorGuard()
    if key == "g3":
        from .claude_guard import ClaudeGuard
        return ClaudeGuard(store, tracker=tracker, client=client)
    if key == "g4":
        from .cascade import CascadeGuard
        from .claude_guard import ClaudeGuard
        from .detector_guard import DetectorGuard
        return CascadeGuard(DetectorGuard(), ClaudeGuard(store, tracker=tracker, client=client))
    if key not in GUARDS:
        raise KeyError(f"unknown guard {key!r}; available: {sorted(GUARDS) + ['g2', 'g3', 'g4']}")
    return GUARDS[key]()
