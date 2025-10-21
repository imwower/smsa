"""Spiking neural network building blocks for SMSA."""

from .lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from .dense import DenseLIF

__all__ = [
    "LIFParams",
    "triangular_surrogate",
    "fast_sigmoid_surrogate",
    "DenseLIF",
]
