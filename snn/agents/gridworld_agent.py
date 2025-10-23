"""GridWorld-specific spiking policy agent utilities."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from meta.autoadapt import CodePatcher
from snn.agents.base import PolicyState
from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate


Action = int  # 0: 上, 1: 下, 2: 左, 3: 右


def patched_surrogate(u: float, gain: float = 1.5) -> float:
    """Code patch surrogate: narrower gradient window for sharper spikes."""
    denom = 1.0 + gain * u * u
    return gain / (denom * denom)


def _softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _spike_from_rate(rate: float) -> int:
    rate = max(0.0, min(1.0, rate))
    return 1 if random.random() < rate else 0


def _clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def build_self_model_input(
    *,
    state_index: int,
    action: int,
    mean_rate: float,
    sum_rate: float,
    eta_e: float,
    v_th: float,
    state_size: int,
) -> List[float]:
    """Compose Self-Model input features from policy signals."""
    obs_vec = [0.0 for _ in range(state_size)]
    obs_vec[state_index] = 1.0
    action_vec = [0.0 for _ in range(4)]
    action_vec[action] = 1.0
    extras = [
        max(0.0, min(mean_rate, 1.0)),
        max(0.0, min(sum_rate, 1.0)),
        max(0.0, min(eta_e, 1.0)),
        max(0.0, min(v_th, 1.0)),
    ]
    return obs_vec + action_vec + extras


class GridworldAgent:
    """DenseLIF policy agent with intrinsic reward and meta hooks."""

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
        intrinsic_beta: float = 0.3,
        temporal_state_size: int | None = None,
        temporal_beta: float = 0.85,
    ) -> None:
        self.hidden = DenseLIF(
            n_in=state_size + (temporal_state_size or state_size),
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
        self.state_size = state_size
        self.intrinsic_beta = intrinsic_beta
        self.surrogate_name = "fast_sigmoid"
        self.patched = False
        self.code_patcher = CodePatcher()
        self.last_patch_info: str | None = None
        self.temporal_state_size = temporal_state_size or state_size
        self.temporal_unit = LinearTemporalUnit(
            n_in=state_size,
            n_state=self.temporal_state_size,
            beta=temporal_beta,
        )

    def intrinsic_bonus(self, visit_count: int) -> float:
        return self.intrinsic_beta / math.sqrt(visit_count + 1)

    def encode_state(self, index: int) -> List[int]:
        vec = [0 for _ in range(self.state_size)]
        vec[index] = 1
        return vec

    def begin_episode(self) -> None:
        self.temporal_unit.reset()

    def forward(self, state_index: int) -> PolicyState:
        bits = self.encode_state(state_index)
        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]
        eligibility_history: List[List[List[float]]] = []
        bias_history: List[List[float]] = []
        for _ in range(self.inner_steps):
            base_rates = [
                self.high_rate if bit else self.low_rate for bit in bits
            ]
            temporal_state = self.temporal_unit.transform(
                [float(bit) for bit in bits]
            )
            temporal_rates = [max(0.0, min(rate, 1.0)) for rate in temporal_state]
            combined_rates = base_rates + temporal_rates
            pre_spikes = [_spike_from_rate(rate) for rate in combined_rates]
            spikes, _, eligibility_snapshot, bias_snapshot = self.hidden.step(
                pre_spikes
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

    def set_surrogate(self, name: str) -> None:
        if name == "triangular":
            self.hidden.set_surrogate(triangular_surrogate)
        elif name == "patched":
            self.hidden.set_surrogate(patched_surrogate)
        else:
            self.hidden.set_surrogate(fast_sigmoid_surrogate)
        self.surrogate_name = name

    def add_neuron(self) -> bool:
        for row in self.hidden.weights:
            row.append(random.uniform(-0.2, 0.2))
        self.hidden.bias.append(0.0)
        for row in self.hidden.eligibility:
            row.append(0.0)
        self.hidden.bias_eligibility.append(0.0)
        self.hidden.n_out += 1
        self.hidden.reset_state()
        self.readout_weights.append([random.uniform(-0.2, 0.2) for _ in range(4)])
        return True

    def prune_neuron(self) -> bool:
        if self.hidden.n_out <= 6:
            return False
        idx = self.hidden.n_out - 1
        for row in self.hidden.weights:
            row.pop(idx)
        self.hidden.bias.pop(idx)
        for row in self.hidden.eligibility:
            row.pop(idx)
        self.hidden.bias_eligibility.pop(idx)
        self.hidden.n_out -= 1
        self.hidden.reset_state()
        self.readout_weights.pop(idx)
        return True

    def apply_modification(self, action: str) -> Tuple[bool, str | None]:
        translations = {
            "inner_up": "inner_steps_up",
            "inner_down": "inner_steps_down",
            "patch_surrogate": "code_patch_surrogate",
            "vth_up": "v_th_up",
            "vth_down": "v_th_down",
        }
        action = translations.get(action, action)
        info: str | None = None
        if action == "eta_up":
            self.hidden_lr = min(self.hidden_lr * 1.2, 0.6)
            self.baseline_beta = min(self.baseline_beta * 1.1, 0.2)
            return True, info
        if action == "eta_down":
            self.hidden_lr = max(self.hidden_lr * 0.8, 0.05)
            self.baseline_beta = max(self.baseline_beta * 0.9, 0.02)
            return True, info
        if action == "v_th_up":
            self.hidden.params.v_th = min(self.hidden.params.v_th + 0.05, 1.2)
            return True, info
        if action == "v_th_down":
            self.hidden.params.v_th = max(self.hidden.params.v_th - 0.05, 0.2)
            return True, info
        if action == "intrinsic_up":
            self.intrinsic_beta = min(self.intrinsic_beta * 1.2, 0.8)
            return True, info
        if action == "intrinsic_down":
            self.intrinsic_beta = max(self.intrinsic_beta * 0.8, 0.05)
            return True, info
        if action == "inner_steps_up":
            self.inner_steps = min(self.inner_steps + 4, 36)
            return True, info
        if action == "inner_steps_down":
            if self.inner_steps <= 10:
                return False, info
            self.inner_steps = max(self.inner_steps - 4, 8)
            return True, info
        if action == "add_neuron":
            added = self.add_neuron()
            if added:
                info = f"n_hidden={self.hidden.n_out}"
            return added, info
        if action == "prune_neuron":
            pruned = self.prune_neuron()
            if pruned:
                info = f"n_hidden={self.hidden.n_out}"
            return pruned, info
        if action == "switch_surrogate":
            next_name = (
                "triangular"
                if self.surrogate_name == "fast_sigmoid"
                else "fast_sigmoid"
            )
            self.set_surrogate(next_name)
            return True, info
        if action == "code_patch_surrogate":
            info = self.code_patcher.apply(self.hidden)
            self.patched = True
            self.surrogate_name = "dynamic_patch"
            self.last_patch_info = info
            return True, info
        return False, info


__all__ = [
    "Action",
    "GridworldAgent",
    "build_self_model_input",
    "patched_surrogate",
]
