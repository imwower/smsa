"""Leaky Integrate-and-Fire neuron utilities and surrogate gradients."""

from dataclasses import dataclass
import math
from typing import Callable


SurrogateFn = Callable[[float], float]


@dataclass
class LIFParams:
    """Container for LIF neuron hyperparameters."""

    v_th: float
    tau_m: float
    tau_a: float
    beta: float
    refractory: int


def triangular_surrogate(u: float, width: float = 1.0) -> float:
    """Symmetric triangular surrogate derivative."""
    if u >= width or u <= -width:
        return 0.0
    return (width - abs(u)) / width


def fast_sigmoid_surrogate(u: float, slope: float = 2.0) -> float:
    """Fast sigmoid surrogate derivative."""
    denom = 1.0 + slope * abs(u)
    return slope / (denom * denom)
