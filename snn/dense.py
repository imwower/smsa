"""带资格迹的 LIF 稠密层（e-prop 三因子）。"""

from typing import Callable, List, Sequence, Tuple
import math
import random

from .lif import LIFParams, SurrogateFn


class DenseLIF:
    """维护逐突触资格迹的 LIF 稠密群体。"""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        params: LIFParams,
        surrogate_fn: SurrogateFn,
        eligibility_lambda: float | None = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.params = params
        self.set_surrogate(surrogate_fn)
        self.weights = [
            [random.uniform(-0.5, 0.5) for _ in range(n_out)] for _ in range(n_in)
        ]
        self.bias = [random.uniform(-0.1, 0.1) for _ in range(n_out)]
        if eligibility_lambda is None:
            self.eligibility_lambda = math.exp(-1.0 / max(params.tau_m, 1.0))
        else:
            self.eligibility_lambda = eligibility_lambda
        # 预缓存衰减与倒数，减少除法与exp开销
        self._inv_tau_m = 1.0 / max(self.params.tau_m, 1e-6)
        self._inv_tau_a = 1.0 / max(self.params.tau_a, 1e-6)
        self._profile_on = False
        self._profile_n = 0
        self._profile_acc = 0.0
        self.reset_state()

    def set_surrogate(self, surrogate_fn: SurrogateFn) -> None:
        self.surrogate_fn = surrogate_fn

    def reset_state(self) -> None:
        self.v = [0.0 for _ in range(self.n_out)]
        self.a = [0.0 for _ in range(self.n_out)]
        self.refractory = [0 for _ in range(self.n_out)]
        self.eligibility = [
            [0.0 for _ in range(self.n_out)] for _ in range(self.n_in)
        ]
        self.bias_eligibility = [0.0 for _ in range(self.n_out)]

    def step(
        self, pre_spikes: Sequence[int]
    ) -> Tuple[List[int], List[float], List[List[float]], List[float]]:
        # 简易性能分析：每 1000 步打印均值（关闭默认开关）
        _t0 = None
        if self._profile_on:
            import time as _t
            _t0 = _t.perf_counter()
        spikes = [0 for _ in range(self.n_out)]
        psis = [0.0 for _ in range(self.n_out)]
        # 复用快照缓冲，降低临时分配
        eligibility_snapshot = [
            [0.0 for _ in range(self.n_out)] for _ in range(self.n_in)
        ]
        bias_snapshot = [0.0 for _ in range(self.n_out)]
        for j in range(self.n_out):
            for i in range(self.n_in):
                self.eligibility[i][j] *= self.eligibility_lambda
            self.bias_eligibility[j] *= self.eligibility_lambda
            psi = 0.0
            if self.refractory[j] > 0:
                self.refractory[j] -= 1
                self.v[j] = 0.0
                self.a[j] += (-self.a[j]) * self._inv_tau_a
            else:
                input_current = self.bias[j]
                for i in range(self.n_in):
                    input_current += self.weights[i][j] * pre_spikes[i]
                dv = (-self.v[j] + input_current - self.params.beta * self.a[j]) * self._inv_tau_m
                v_new = self.v[j] + dv
                u = v_new - self.params.v_th
                psi = self.surrogate_fn(u)
                fired = 1 if v_new >= self.params.v_th else 0
                spikes[j] = fired
                if fired:
                    self.v[j] = 0.0
                    self.refractory[j] = self.params.refractory
                else:
                    self.v[j] = v_new
                self.a[j] += (-self.a[j] + fired) * self._inv_tau_a
            for i in range(self.n_in):
                self.eligibility[i][j] += psi * pre_spikes[i]
                eligibility_snapshot[i][j] = self.eligibility[i][j]
            self.bias_eligibility[j] += psi
            bias_snapshot[j] = self.bias_eligibility[j]
            psis[j] = psi
        if self._profile_on and _t0 is not None:
            import time as _t
            self._profile_n += 1
            self._profile_acc += (_t.perf_counter() - _t0)
            if self._profile_n % 1000 == 0:
                avg = self._profile_acc / float(max(1, self._profile_n))
                print(f"[DenseLIF] 平均 step 开销 {avg*1e3:.3f} ms （{self._profile_n} 步）")
        return spikes, psis, eligibility_snapshot, bias_snapshot

    def eprop_apply(self, learning_signal: Sequence[float], lr: float) -> None:
        """根据学习信号与资格迹更新权重与偏置。"""
        if len(learning_signal) != self.n_out:
            raise ValueError("学习信号长度必须等于输出神经元数量。")
        if lr <= 0.0:
            raise ValueError("学习率必须为正数。")
        # 本地双层循环，尽量减少索引开销
        for i in range(self.n_in):
            row = self.weights[i]
            eli = self.eligibility[i]
            for j in range(self.n_out):
                row[j] -= lr * learning_signal[j] * eli[j]
        for j in range(self.n_out):
            self.bias[j] -= lr * learning_signal[j] * self.bias_eligibility[j]

    # --- AutoGrow：动态扩容/裁剪 -------------------------------------------
    def grow_hidden(self, k: int) -> None:
        """增加 k 个隐藏神经元，扩展权重/偏置/状态/资格迹。"""
        k = max(0, int(k))
        if k == 0:
            return
        for i in range(self.n_in):
            row = self.weights[i]
            eli = self.eligibility[i]
            for _ in range(k):
                row.append(random.uniform(-0.1, 0.1))
                eli.append(0.0)
        for _ in range(k):
            self.bias.append(0.0)
            self.bias_eligibility.append(0.0)
            self.v.append(0.0)
            self.a.append(0.0)
            self.refractory.append(0)
        self.n_out += k

    def prune_hidden(self, idxs: Sequence[int]) -> None:
        """按列裁剪隐藏神经元（多列）。"""
        cols = sorted({int(i) for i in idxs if 0 <= int(i) < self.n_out}, reverse=True)
        if not cols:
            return
        for i in range(self.n_in):
            row = self.weights[i]
            eli = self.eligibility[i]
            for c in cols:
                if 0 <= c < len(row):
                    row.pop(c)
                if 0 <= c < len(eli):
                    eli.pop(c)
        for c in cols:
            if 0 <= c < len(self.bias):
                self.bias.pop(c)
            if 0 <= c < len(self.bias_eligibility):
                self.bias_eligibility.pop(c)
            if 0 <= c < len(self.v):
                self.v.pop(c)
            if 0 <= c < len(self.a):
                self.a.pop(c)
            if 0 <= c < len(self.refractory):
                self.refractory.pop(c)
        self.n_out = max(0, self.n_out - len(cols))

    # 调试辅助：启用/禁用 profile
    def profile_step(self, on: bool = False) -> None:
        self._profile_on = bool(on)


class LinearTemporalUnit:
    """线性时序寄存器：s_t = α s_{t-1} + (1-α) f(x_t)。"""

    def __init__(
        self,
        n_in: int,
        n_state: int,
        beta: float = 0.85,
    ) -> None:
        self.n_in = n_in
        self.n_state = n_state
        self.beta = beta
        self.weights = [
            [0.0 for _ in range(n_state)]
            for _ in range(n_in)
        ]
        self.bias = [0.0 for _ in range(n_state)]
        self.gate_weights = [random.uniform(-0.2, 0.2) for _ in range(n_in)]
        self.gate_bias = 0.0
        self.reset()
        self._init_identity()

    def reset(self) -> None:
        self.state = [0.0 for _ in range(self.n_state)]
        self.gate_trace = 0.0

    def _init_identity(self) -> None:
        shared = min(self.n_in, self.n_state)
        for i in range(self.n_in):
            for j in range(self.n_state):
                self.weights[i][j] = 0.0
        for idx in range(shared):
            self.weights[idx][idx] = 1.0

    def transform(self, x: Sequence[float]) -> List[float]:
        self.gate_trace = (
            self.beta * self.gate_trace
            + (1.0 - self.beta) * (sum(x) / float(max(len(x), 1)))
        )
        gate_input = self.gate_bias
        for w, xv in zip(self.gate_weights, x):
            gate_input += w * xv
        gate_input += self.gate_trace
        alpha = 1.0 / (1.0 + math.exp(-gate_input))
        alpha = max(0.05, min(0.95, alpha))
        f_vals: List[float] = []
        for j in range(self.n_state):
            acc = self.bias[j]
            for i in range(self.n_in):
                acc += self.weights[i][j] * x[i]
            f_vals.append(acc)
        self.state = [
            alpha * self.state[j] + (1.0 - alpha) * f_vals[j]
            for j in range(self.n_state)
        ]
        return self.state[:]

    def reinit(self) -> None:
        self.weights = [
            [0.0 for _ in range(self.n_state)]
            for _ in range(self.n_in)
        ]
        self.bias = [0.0 for _ in range(self.n_state)]
        self.gate_weights = [random.uniform(-0.2, 0.2) for _ in range(self.n_in)]
        self.gate_bias = 0.0
        self.reset()
        self._init_identity()
