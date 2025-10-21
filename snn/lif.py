"""泄露积分放电（LIF）神经元工具与替代导数。"""

from dataclasses import dataclass
import math
from typing import Callable


SurrogateFn = Callable[[float], float]


@dataclass
class LIFParams:
    """LIF 神经元超参数容器。"""

    v_th: float
    tau_m: float
    tau_a: float
    beta: float
    refractory: int


def triangular_surrogate(u: float, width: float = 1.0) -> float:
    """对称三角形替代导数。"""
    if u >= width or u <= -width:
        return 0.0
    return (width - abs(u)) / width


def fast_sigmoid_surrogate(u: float, slope: float = 2.0) -> float:
    """快速 Sigmoid 替代导数。"""
    denom = 1.0 + slope * abs(u)
    return slope / (denom * denom)
