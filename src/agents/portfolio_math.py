# src/agents/portfolio_math.py
from __future__ import annotations

from typing import Dict, List, Literal


# Risk buckets and target weights


RiskLevel = Literal["low", "medium", "high"]

# Target bucket weights by risk level.
# You can tweak these numbers later if you want.
RISK_BUCKETS: Dict[RiskLevel, Dict[str, float]] = {
    "low": {
        "defensive": 0.60,
        "core_growth": 0.35,
        "satellite": 0.05,
    },
    "medium": {
        "defensive": 0.30,
        "core_growth": 0.50,
        "satellite": 0.20,
    },
    "high": {
        "defensive": 0.10,
        "core_growth": 0.45,
        "satellite": 0.45,
    },
}


def compute_portfolio_weights(
    risk_tolerance: str,
    bucket_assignment: Dict[str, str],
    max_per_stock: float = 0.25,
) -> Dict[str, float]:
    """
    Given the user's risk level and a mapping of tickers to risk buckets,
    compute target portfolio weights per ticker.

    Args:
        risk_tolerance: "low" | "medium" | "high" (case-insensitive).
        bucket_assignment: mapping ticker -> "defensive" | "core_growth" | "satellite".
        max_per_stock: hard cap per single ticker (e.g. 0.25 for 25%).

    Returns:
        Dict[ticker, weight_fraction] where weights sum to ~1.0.
    """
    # Normalize risk level and fall back to "medium" if weird
    rt_str = (risk_tolerance or "medium").lower()
    if rt_str not in RISK_BUCKETS:
        rt_str = "medium"
    rt: RiskLevel = rt_str  # type: ignore[assignment]

    bucket_targets = RISK_BUCKETS[rt]

    # Group tickers by bucket
    by_bucket: Dict[str, List[str]] = {
        "defensive": [],
        "core_growth": [],
        "satellite": [],
    }
    for ticker, bucket in bucket_assignment.items():
        b = bucket if bucket in by_bucket else "core_growth"
        by_bucket[b].append(ticker)

    # Initial per-ticker weights based purely on bucket targets
    weights: Dict[str, float] = {}

    for bucket, tickers in by_bucket.items():
        if not tickers:
            continue
        target_bucket_weight = bucket_targets[bucket]
        per_ticker = target_bucket_weight / len(tickers)
        for t in tickers:
            weights[t] = per_ticker

    # Enforce max_per_stock cap and renormalize
    capped_total = 0.0
    for t in list(weights.keys()):
        if weights[t] > max_per_stock:
            weights[t] = max_per_stock
        capped_total += weights[t]

    if capped_total <= 0:
        return {}

    # Renormalize so everything sums to 1.0
    for t in weights:
        weights[t] = weights[t] / capped_total

    return weights

