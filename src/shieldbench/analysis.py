"""Extra analysis used by the dashboard Analysis tab and the datasheet:
 - mcnemar_exact: paired test, did guard B change the outcome of the SAME attacks compared with guard A?
 - pairwise_tests: that test for the pairs that matter
 - tradeoff_table: one row per guard with attack success, harmless blocked, mean delay (for the cost-safety chart)
 - mutation_heatmap_data: attack success by disguise and guard
 - rogan_gladen: correct an observed attack-success rate for a judge's known error rates"""
from math import comb
import pandas as pd


def mcnemar_exact(a_success, b_success) -> dict:
    """a_success and b_success: 0/1 per attack, same order. Exact two-sided binomial test on the discordant pairs.
    b_only = attacks that worked against A but not against B; a_only = the reverse."""
    a_only = sum(1 for x, y in zip(a_success, b_success) if x == 0 and y == 1)
    b_only = sum(1 for x, y in zip(a_success, b_success) if x == 1 and y == 0)
    n = a_only + b_only
    if n == 0:
        return {"worked_only_vs_A": b_only, "worked_only_vs_B": a_only, "p_value": 1.0}
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"worked_only_vs_A": b_only, "worked_only_vs_B": a_only, "p_value": round(p, 4)}


def _attack_outcomes(df: pd.DataFrame, col: str = "verdict") -> pd.DataFrame:
    a = df[df["benign"] == 0]
    a = a.assign(_k=a["attack_id"].astype(str) + "|" + (a["mutation"].fillna("none").astype(str) if "mutation" in a else "none"))
    return a.pivot_table(index="_k", columns="guard", values=col, aggfunc="max")


def pairwise_tests(df: pd.DataFrame, pairs=None, col: str = "verdict") -> pd.DataFrame:
    """McNemar for guard pairs on attacks both guards were run on. pairs: list of (guardA, guardB) names or short keys like 'g0'."""
    out = _attack_outcomes(df, col)
    names = {c.split("_")[0]: c for c in out.columns}
    pairs = pairs or [("g0", "g1"), ("g0", "g2"), ("g0", "g3"), ("g0", "g4"), ("g1", "g4"), ("g3", "g4"), ("g2", "g4")]
    rows = []
    for ka, kb in pairs:
        a, b = names.get(ka, ka), names.get(kb, kb)
        if a not in out.columns or b not in out.columns:
            continue
        both = out[[a, b]].dropna()
        r = mcnemar_exact(both[a].astype(int).tolist(), both[b].astype(int).tolist())
        rows.append({"A": a, "B": b, "attacks": len(both), "worked on A, not B": r["worked_only_vs_A"],
                     "worked on B, not A": r["worked_only_vs_B"], "p value": r["p_value"],
                     "reading": "different" if r["p_value"] < 0.05 else "no clear difference"})
    return pd.DataFrame(rows)


def tradeoff_table(df: pd.DataFrame, col: str = "verdict") -> pd.DataFrame:
    rows = []
    for g, grp in df.groupby("guard"):
        atk, ben = grp[grp["benign"] == 0], grp[grp["benign"] == 1]
        ms = float(grp["guard_s"].mean() * 1000) if "guard_s" in grp and grp["guard_s"].notna().any() else float("nan")
        rows.append({"guard": g, "attack success %": round(float(atk[col].mean()) * 100, 1) if len(atk) else float("nan"),
                     "harmless blocked %": round(float(ben["blocked"].mean()) * 100, 1) if len(ben) else float("nan"),
                     "mean guard delay (ms)": round(ms, 2)})
    return pd.DataFrame(rows)


def mutation_heatmap_data(df: pd.DataFrame, col: str = "verdict") -> pd.DataFrame:
    a = df[df["benign"] == 0].assign(mutation=lambda d: d["mutation"].fillna("none"))
    t = a.groupby(["mutation", "guard"])[col].agg(["mean", "size"]).reset_index()
    t["attack success %"] = (t["mean"] * 100).round(1)
    return t.rename(columns={"size": "attacks"})[["mutation", "guard", "attack success %", "attacks"]]


def rogan_gladen(observed: float, sensitivity: float, specificity: float) -> float:
    """True rate = (observed + specificity - 1) / (sensitivity + specificity - 1), clipped to 0..1.
    sensitivity: share of real leaks the judge flags; specificity: share of non-leaks it clears."""
    d = sensitivity + specificity - 1
    if d <= 0:
        return float("nan")
    return min(1.0, max(0.0, (observed + specificity - 1) / d))
