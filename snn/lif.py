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


# AUTOPATCH SURROGATE START
# 使用矩形窗的替代导数；窗口宽度 ~ 1/slope
def fast_sigmoid_surrogate(u: float, slope: float = 2.0) -> float:
    """矩形窗替代导数：在小邻域给常数梯度。"""
    width = 1.0 / max(1e-6, slope)
    return (slope if -width <= u <= width else 0.0)
# AUTOPATCH SURROGATE END


# 工厂：生成带固定参数的替代导数闭包，避免重复计算常量
def make_rect_surrogate(*, slope: float = 2.0) -> SurrogateFn:
    width = 1.0 / max(1e-6, slope)
    def fn(u: float) -> float:
        return slope if -width <= u <= width else 0.0
    return fn


def make_triangular_surrogate(*, width: float = 1.0) -> SurrogateFn:
    w = max(1e-6, float(width))
    def fn(u: float) -> float:
        if u >= w or u <= -w:
            return 0.0
        return (w - abs(u)) / w
    return fn
