"""Self-Model assisted spiking policy for SMSA training loops."""

from __future__ import annotations

import math
import random
from typing import List, Sequence

from snn.agents.base import PolicyState
from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate


def _softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _sample_poisson(bit: int, high_rate: float, low_rate: float) -> float:
    rate = high_rate if bit else low_rate
    return 1.0 if random.random() < rate else 0.0


def build_self_model_input(
    *,
    state_index: int,
    action: int,
    mean_count: float,
    sum_count: float,
    eta_e: float,
    v_th: float,
    state_size: int,
) -> List[float]:
    obs_vec = [0.0 for _ in range(state_size)]
    obs_vec[state_index] = 1.0
    action_vec = [0.0 for _ in range(4)]
    action_vec[action] = 1.0
    extras = [
        max(0.0, min(mean_count, 1.0)),
        max(0.0, min(sum_count, 1.0)),
        max(0.0, min(eta_e, 1.0)),
        max(0.0, min(v_th, 1.0)),
    ]
    return obs_vec + action_vec + extras


class SNNPolicy:
    """Spiking policy network used in SMSA training loops."""

    def __init__(
        self,
        *,
        state_size: int,
        hidden_size: int,
        params: LIFParams,
        inner_steps: int = 18,
        high_rate: float = 0.95,
        low_rate: float = 0.05,
        hidden_lr: float = 0.15,
        readout_lr: float = 0.3,
        clip: float = 2.0,
        intrinsic_beta: float = 0.2,
        use_temporal_unit: bool = False,
        temporal_state_size: int | None = None,
    ) -> None:
        self.state_size = state_size
        self.use_temporal_unit = use_temporal_unit
        self.temporal_state_size = (
            temporal_state_size if temporal_state_size is not None else state_size
        )
        input_dim = (
            state_size + self.temporal_state_size if use_temporal_unit else state_size
        )
        self.temporal_unit = (
            LinearTemporalUnit(state_size, self.temporal_state_size)
            if use_temporal_unit
            else None
        )
        self.hidden = DenseLIF(
            n_in=input_dim,
            n_out=hidden_size,
            params=params,
            surrogate_fn=fast_sigmoid_surrogate,
        )
        self.readout_weights = [
            [random.uniform(-0.2, 0.2) for _ in range(4)]
            for _ in range(hidden_size)
        ]
        self.readout_bias = [0.0 for _ in range(4)]
        self.inner_steps = inner_steps
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.hidden_lr = hidden_lr
        self.readout_lr = readout_lr
        self.clip = clip
        self.baseline = 0.0
        self.baseline_beta = 0.05
        self.intrinsic_beta = intrinsic_beta

    def encode_state(self, index: int) -> List[int]:
        vec = [0 for _ in range(self.state_size)]
        vec[index] = 1
        return vec

    def forward(self, state_index: int) -> PolicyState:
        bits = self.encode_state(state_index)
        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]
        eligibility_history: List[List[List[float]]] = []
        bias_history: List[List[float]] = []
        for _ in range(self.inner_steps):
            base_input = [
                _sample_poisson(bit, self.high_rate, self.low_rate)
                for bit in bits
            ]
            if self.temporal_unit is not None:
                temporal_state = self.temporal_unit.transform(base_input)
                combined_input = base_input + temporal_state
            else:
                combined_input = base_input
            spikes, _, eligibility_snapshot, bias_snapshot = self.hidden.step(
                combined_input
            )
            hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]
            eligibility_history.append([row[:] for row in eligibility_snapshot])
            bias_history.append(bias_snapshot[:])
        hidden_rates = [count / float(self.inner_steps) for count in hidden_counts]
        logits = []
        for action in range(4):
            logit = self.readout_bias[action]
            for h in range(self.hidden.n_out):
                logit += self.readout_weights[h][action] * hidden_rates[h]
            logits.append(logit)
        probs = _softmax(logits)
        total_spikes = sum(hidden_counts)
        mean_rate = (total_spikes / max(self.hidden.n_out, 1)) / float(self.inner_steps)
        sum_rate = min(total_spikes / float(self.inner_steps), 1.0)
        return PolicyState(
            probs=probs,
            eligibility_history=eligibility_history,
            bias_history=bias_history,
            hidden_rates=hidden_rates,
            hidden_counts=hidden_counts,
            mean_rate=min(mean_rate, 1.0),
            sum_rate=sum_rate,
        )

    def sample_action(self, probs: Sequence[float]) -> int:
        threshold = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                return idx
        return len(probs) - 1

    def update_baseline(self, reward: float) -> None:
        self.baseline = (
            (1.0 - self.baseline_beta) * self.baseline
            + self.baseline_beta * reward
        )

    def intrinsic_bonus(self, visit_count: int) -> float:
        return self.intrinsic_beta / math.sqrt(visit_count + 1)

    def update(
        self,
        state: PolicyState,
        action: int,
        advantage: float,
        *,
        self_signal: Sequence[float] | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        policy_error = [p for p in state.probs]
        policy_error[action] -= 1.0
        policy_error = [
            _clip_value(err * advantage, self.clip) for err in policy_error
        ]
        learning_signals = []
        for h in range(self.hidden.n_out):
            signal = 0.0
            for a in range(4):
                signal += policy_error[a] * self.readout_weights[h][a]
            learning_signals.append(_clip_value(signal, self.clip))
        if self_signal is not None:
            if len(self_signal) != self.hidden.n_out:
                adjusted = [0.0 for _ in range(self.hidden.n_out)]
                limit = min(len(self_signal), self.hidden.n_out)
                for h in range(limit):
                    adjusted[h] = self_signal[h]
                self_signal = adjusted
            combined = []
            for h in range(self.hidden.n_out):
                combined_signal = alpha * learning_signals[h] + beta * self_signal[h]
                combined.append(_clip_value(combined_signal, self.clip))
            learning_signals = combined
        else:
            learning_signals = [_clip_value(alpha * sig, self.clip) for sig in learning_signals]
        for i in range(self.hidden.n_in):
            for h in range(self.hidden.n_out):
                grad = 0.0
                for elig in state.eligibility_history:
                    grad += learning_signals[h] * elig[i][h]
                grad = _clip_value(grad, self.clip)
                self.hidden.weights[i][h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            grad = 0.0
            for bias_elig in state.bias_history:
                grad += learning_signals[h] * bias_elig[h]
            grad = _clip_value(grad, self.clip)
            self.hidden.bias[h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            for a in range(4):
                grad = _clip_value(policy_error[a] * state.hidden_rates[h], self.clip)
                self.readout_weights[h][a] -= self.readout_lr * grad
        for a in range(4):
            grad = _clip_value(policy_error[a], self.clip)
            self.readout_bias[a] -= self.readout_lr * grad

    def reset_parameters(self) -> None:
        for i in range(self.hidden.n_in):
            self.hidden.weights[i] = [
                random.uniform(-0.2, 0.2) for _ in range(self.hidden.n_out)
            ]
        self.hidden.bias = [0.0 for _ in range(self.hidden.n_out)]
        self.hidden.reset_state()
        self.readout_weights = [
            [random.uniform(-0.2, 0.2) for _ in range(4)]
            for _ in range(self.hidden.n_out)
        ]
        self.readout_bias = [0.0 for _ in range(4)]
        self.baseline = 0.0
        if self.temporal_unit is not None:
            self.temporal_unit.reinit()

    def begin_episode(self) -> None:
        if self.temporal_unit is not None:
            self.temporal_unit.reset()


__all__ = ["SNNPolicy", "build_self_model_input"]
