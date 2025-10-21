"""Dense layer of LIF neurons with eligibility traces for e-prop."""

from typing import Callable, List, Sequence, Tuple
import math
import random

from .lif import LIFParams, SurrogateFn


class DenseLIF:
    """Dense LIF population maintaining per-synapse eligibility traces."""

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
        spikes = [0 for _ in range(self.n_out)]
        psis = [0.0 for _ in range(self.n_out)]
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
                self.a[j] += (-self.a[j]) / self.params.tau_a
            else:
                input_current = self.bias[j]
                for i in range(self.n_in):
                    input_current += self.weights[i][j] * pre_spikes[i]
                dv = (
                    -self.v[j]
                    + input_current
                    - self.params.beta * self.a[j]
                ) / self.params.tau_m
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
                self.a[j] += (-self.a[j] + fired) / self.params.tau_a
            for i in range(self.n_in):
                self.eligibility[i][j] += psi * pre_spikes[i]
                eligibility_snapshot[i][j] = self.eligibility[i][j]
            self.bias_eligibility[j] += psi
            bias_snapshot[j] = self.bias_eligibility[j]
            psis[j] = psi
        return spikes, psis, eligibility_snapshot, bias_snapshot
