class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    """Hard spending cap. Cached calls cost nothing, so reruns are free."""

    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.spent = 0.0

    def add(self, usd: float):
        self.spent += usd
        if self.spent > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} > cap ${self.cap:.2f}; stopping the run")
