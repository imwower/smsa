"""Common policy-state dataclasses shared by SNN agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PolicyState:
    """Snapshot of policy head state used for e-prop style updates."""

    probs: List[float]
    eligibility_history: List[List[List[float]]]
    bias_history: List[List[float]]
    hidden_rates: List[float]
    hidden_counts: List[int]
    mean_rate: float
    sum_rate: float


__all__ = ["PolicyState"]
