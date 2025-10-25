"""Self-Model 模块：预测下一观察、奖励、能耗与自因概率。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate


def _clip(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _spike_from_rate(rate: float, rng: random.Random) -> int:
    rate = max(0.0, min(1.0, rate))
    return 1 if rng.random() < rate else 0


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [v / total for v in exps]


@dataclass
class SelfModelState:
    """前向传播缓存，用于自我模型更新。"""

    probs_next: List[float]
    pred_reward: float
    pred_energy: float
    probs_cause: List[float]
    hidden_rates: List[float]
    hidden_counts: List[int]
    energy_norm: float


class SelfModel:
    """并行预测 GridWorld 下一步状态与自因概率的 Self-Model。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        lif_params: LIFParams,
        *,
        inner_steps: int = 16,
        hidden_lr: float = 0.12,
        readout_lr: float = 0.2,
        clip: float = 2.0,
        obs_weight: float = 1.2,
        reward_weight: float = 0.35,
        energy_weight: float = 0.35,
        cause_weight: float = 0.5,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.extra_dim = 4
        input_size = obs_dim + action_dim + self.extra_dim

        self.hidden = DenseLIF(
            n_in=input_size,
            n_out=hidden_size,
            params=lif_params,
            surrogate_fn=fast_sigmoid_surrogate,
        )
        self.inner_steps = inner_steps
        self.hidden_lr = hidden_lr
        self.readout_lr = readout_lr
        self.clip = clip
        self.obs_weight = obs_weight
        self.reward_weight = reward_weight
        self.energy_weight = energy_weight
        self.cause_weight = cause_weight

        self.next_weights = [
            [random.uniform(-0.15, 0.15) for _ in range(self.obs_dim)]
            for _ in range(hidden_size)
        ]
        self.next_bias = [0.0 for _ in range(self.obs_dim)]

        self.reward_weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.reward_bias = 0.0

        self.energy_weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.energy_bias = 0.0

        self.cause_dim = 2
        self.cause_weights = [
            [random.uniform(-0.15, 0.15) for _ in range(self.cause_dim)]
            for _ in range(hidden_size)
        ]
        self.cause_bias = [0.0 for _ in range(self.cause_dim)]

        self._rng = random.Random(1234)

    def forward(self, features: Sequence[float]) -> SelfModelState:
        """前向传播，返回预测结果与缓存状态。"""
        if len(features) != self.obs_dim + self.action_dim + self.extra_dim:
            raise ValueError("SelfModel 输入维度不匹配。")

        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]

        for _ in range(self.inner_steps):
            spikes_in = [
                _spike_from_rate(rate, self._rng) for rate in features
            ]
            spikes, _, _, _ = self.hidden.step(spikes_in)
            hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]

        hidden_rates = [count / float(self.inner_steps) for count in hidden_counts]

        logits_next = []
        for idx in range(self.obs_dim):
            logit = self.next_bias[idx]
            for h in range(self.hidden.n_out):
                logit += self.next_weights[h][idx] * hidden_rates[h]
            logits_next.append(logit)
        probs_next = _softmax(logits_next)

        pred_reward = self.reward_bias
        pred_energy = self.energy_bias
        for h in range(self.hidden.n_out):
            pred_reward += self.reward_weights[h] * hidden_rates[h]
            pred_energy += self.energy_weights[h] * hidden_rates[h]

        logits_cause = []
        for idx in range(self.cause_dim):
            logit = self.cause_bias[idx]
            for h in range(self.hidden.n_out):
                logit += self.cause_weights[h][idx] * hidden_rates[h]
            logits_cause.append(logit)
        probs_cause = _softmax(logits_cause)

        energy_norm = sum(hidden_counts) / float(
            max(1, self.inner_steps * self.hidden.n_out)
        )

        return SelfModelState(
            probs_next=probs_next,
            pred_reward=pred_reward,
            pred_energy=pred_energy,
            probs_cause=probs_cause,
            hidden_rates=hidden_rates,
            hidden_counts=hidden_counts,
            energy_norm=energy_norm,
        )

    def _head_gradients(
        self,
        *,
        state: SelfModelState,
        next_obs_index: int,
        reward_target: float,
        energy_target: float,
        cause_label: int,
    ) -> Tuple[List[float], float, float, List[float]]:
        if not (0 <= next_obs_index < self.obs_dim):
            raise ValueError("next_obs_index 超出范围。")
        if cause_label not in (0, 1):
            raise ValueError("cause_label 必须为 0/1。")

        target_next = [0.0 for _ in range(self.obs_dim)]
        target_next[next_obs_index] = 1.0
        next_grad = [
            self.obs_weight * (state.probs_next[idx] - target_next[idx])
            for idx in range(self.obs_dim)
        ]
        reward_grad = self.reward_weight * (state.pred_reward - reward_target)
        energy_grad = self.energy_weight * (state.pred_energy - energy_target)

        target_cause = [0.0, 0.0]
        target_cause[cause_label] = 1.0
        cause_grad = [
            self.cause_weight * (state.probs_cause[idx] - target_cause[idx])
            for idx in range(self.cause_dim)
        ]
        return next_grad, reward_grad, energy_grad, cause_grad

    def learning_signal(
        self,
        *,
        state: SelfModelState,
        next_grad: Sequence[float],
        reward_grad: float,
        energy_grad: float,
        cause_grad: Sequence[float],
    ) -> List[float]:
        """汇总各读出头梯度并返回 L_self。"""
        signals = [0.0 for _ in range(self.hidden.n_out)]
        for idx, grad in enumerate(next_grad):
            for h in range(self.hidden.n_out):
                signals[h] += grad * self.next_weights[h][idx]
        for h in range(self.hidden.n_out):
            signals[h] += reward_grad * self.reward_weights[h]
            signals[h] += energy_grad * self.energy_weights[h]
        for idx, grad in enumerate(cause_grad):
            for h in range(self.hidden.n_out):
                signals[h] += grad * self.cause_weights[h][idx]
        return [_clip(sig, self.clip) for sig in signals]

    def update_heads(
        self,
        *,
        state: SelfModelState,
        next_grad: Sequence[float],
        reward_grad: float,
        energy_grad: float,
        cause_grad: Sequence[float],
    ) -> None:
        """使用隐藏层平均放电率更新读出层参数。"""
        for h in range(self.hidden.n_out):
            rate = state.hidden_rates[h]
            for idx, grad in enumerate(next_grad):
                delta = _clip(grad * rate, self.clip)
                self.next_weights[h][idx] -= self.readout_lr * delta
            delta_reward = _clip(reward_grad * rate, self.clip)
            self.reward_weights[h] -= self.readout_lr * delta_reward
            delta_energy = _clip(energy_grad * rate, self.clip)
            self.energy_weights[h] -= self.readout_lr * delta_energy
            for idx, grad in enumerate(cause_grad):
                delta_cause = _clip(grad * rate, self.clip)
                self.cause_weights[h][idx] -= self.readout_lr * delta_cause

        for idx, grad in enumerate(next_grad):
            self.next_bias[idx] -= self.readout_lr * _clip(grad, self.clip)
        self.reward_bias -= self.readout_lr * _clip(reward_grad, self.clip)
        self.energy_bias -= self.readout_lr * _clip(energy_grad, self.clip)
        for idx, grad in enumerate(cause_grad):
            self.cause_bias[idx] -= self.readout_lr * _clip(grad, self.clip)

    def update(
        self,
        *,
        state: SelfModelState,
        next_obs_index: int,
        reward_target: float,
        energy_target: float,
        cause_label: int,
    ) -> None:
        """执行一次自我模型参数更新。"""
        (
            next_grad,
            reward_grad,
            energy_grad,
            cause_grad,
        ) = self._head_gradients(
            state=state,
            next_obs_index=next_obs_index,
            reward_target=reward_target,
            energy_target=energy_target,
            cause_label=cause_label,
        )

        l_self = self.learning_signal(
            state=state,
            next_grad=next_grad,
            reward_grad=reward_grad,
            energy_grad=energy_grad,
            cause_grad=cause_grad,
        )
        self.hidden.eprop_apply(l_self, self.hidden_lr)
        self.update_heads(
            state=state,
            next_grad=next_grad,
            reward_grad=reward_grad,
            energy_grad=energy_grad,
            cause_grad=cause_grad,
        )
        return l_self


__all__ = ["SelfModel", "SelfModelState"]
