"""Self-Model 并行训练原型：GridWorld + 策略头 + Self-Model。"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate
from tools.logger import get_logger, setup_logging


Action = int  # 0:上, 1:下, 2:左, 3:右


@dataclass
class GridWorldConfig:
    size: int = 5
    slip: float = 0.05
    max_steps: int = 60
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    goal_reward: float = 1.0
    step_penalty: float = -0.01


class GridWorld:
    def __init__(self, cfg: GridWorldConfig) -> None:
        self.cfg = cfg
        self.position = cfg.start
        self.steps = 0

    def reset(self) -> Tuple[int, int]:
        self.position = self.cfg.start
        self.steps = 0
        return self.position

    def state_index(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cfg.size + pos[1]

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, bool]:
        self.steps += 1
        actual_action = action
        caused_by_self = True
        if random.random() < self.cfg.slip:
            actual_action = random.randint(0, 3)
            caused_by_self = False
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)][actual_action]
        nr = max(0, min(self.cfg.size - 1, self.position[0] + drc[0]))
        nc = max(0, min(self.cfg.size - 1, self.position[1] + drc[1]))
        self.position = (nr, nc)
        reward = self.cfg.step_penalty
        done = False
        if self.position == self.cfg.goal:
            reward += self.cfg.goal_reward
            done = True
        if self.steps >= self.cfg.max_steps:
            done = True
        return self.position, reward, done, caused_by_self


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def spike_from_rate(rate: float) -> int:
    rate = max(0.0, min(1.0, rate))
    return 1 if random.random() < rate else 0


@dataclass
class PolicyState:
    probs: List[float]
    eligibility_history: List[List[List[float]]]
    bias_history: List[List[float]]
    hidden_rates: List[float]
    hidden_counts: List[int]
    mean_rate: float
    sum_rate: float


class SNNPolicy:
    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        params: LIFParams,
        inner_steps: int = 18,
        high_rate: float = 0.95,
        low_rate: float = 0.05,
        hidden_lr: float = 0.15,
        readout_lr: float = 0.3,
        clip: float = 2.0,
    ) -> None:
        self.hidden = DenseLIF(
            n_in=state_size,
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
            pre_spikes = [
                spike_from_rate(self.high_rate if bit else self.low_rate)
                for bit in bits
            ]
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
        probs = softmax(logits)
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

    def update(self, state: PolicyState, action: int, advantage: float) -> None:
        policy_error = [p for p in state.probs]
        policy_error[action] -= 1.0
        policy_error = [
            clip_value(err * advantage, self.clip) for err in policy_error
        ]
        learning_signals = []
        for h in range(self.hidden.n_out):
            signal = 0.0
            for a in range(4):
                signal += policy_error[a] * self.readout_weights[h][a]
            learning_signals.append(clip_value(signal, self.clip))
        for i in range(self.hidden.n_in):
            for h in range(self.hidden.n_out):
                grad = 0.0
                for elig in state.eligibility_history:
                    grad += learning_signals[h] * elig[i][h]
                grad = clip_value(grad, self.clip)
                self.hidden.weights[i][h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            grad = 0.0
            for bias_elig in state.bias_history:
                grad += learning_signals[h] * bias_elig[h]
            grad = clip_value(grad, self.clip)
            self.hidden.bias[h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            for a in range(4):
                grad = clip_value(policy_error[a] * state.hidden_rates[h], self.clip)
                self.readout_weights[h][a] -= self.readout_lr * grad
        for a in range(4):
            grad = clip_value(policy_error[a], self.clip)
            self.readout_bias[a] -= self.readout_lr * grad


@dataclass
class SelfModelState:
    probs_next: List[float]
    probs_cause: List[float]
    pred_reward: float
    pred_energy: float
    eligibility_history: List[List[List[float]]]
    bias_history: List[List[float]]
    hidden_rates: List[float]
    hidden_counts: List[int]
    energy_norm: float


class SelfModel:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        params: LIFParams,
        inner_steps: int = 16,
        hidden_lr: float = 0.12,
        readout_lr: float = 0.2,
        clip: float = 2.0,
    ) -> None:
        self.hidden = DenseLIF(
            n_in=input_size,
            n_out=hidden_size,
            params=params,
            surrogate_fn=fast_sigmoid_surrogate,
        )
        self.inner_steps = inner_steps
        self.hidden_lr = hidden_lr
        self.readout_lr = readout_lr
        self.clip = clip
        self.next_obs_dim = 25
        self.cause_dim = 2
        self.next_weights = [
            [random.uniform(-0.15, 0.15) for _ in range(self.next_obs_dim)]
            for _ in range(hidden_size)
        ]
        self.next_bias = [0.0 for _ in range(self.next_obs_dim)]
        self.reward_weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.reward_bias = 0.0
        self.energy_weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.energy_bias = 0.0
        self.cause_weights = [
            [random.uniform(-0.15, 0.15) for _ in range(self.cause_dim)]
            for _ in range(hidden_size)
        ]
        self.cause_bias = [0.0 for _ in range(self.cause_dim)]
        self.obs_weight = 1.2
        self.reward_weight = 0.35
        self.energy_weight = 0.35
        self.cause_weight = 0.5

    def forward(self, rates: Sequence[float]) -> SelfModelState:
        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]
        eligibility_history: List[List[List[float]]] = []
        bias_history: List[List[float]] = []
        for _ in range(self.inner_steps):
            pre_spikes = [spike_from_rate(rate) for rate in rates]
            spikes, _, eligibility_snapshot, bias_snapshot = self.hidden.step(
                pre_spikes
            )
            hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]
            eligibility_history.append([row[:] for row in eligibility_snapshot])
            bias_history.append(bias_snapshot[:])
        hidden_rates = [count / float(self.inner_steps) for count in hidden_counts]
        logits_next = []
        for idx in range(self.next_obs_dim):
            logit = self.next_bias[idx]
            for h in range(self.hidden.n_out):
                logit += self.next_weights[h][idx] * hidden_rates[h]
            logits_next.append(logit)
        probs_next = softmax(logits_next)
        pred_reward = self.reward_bias
        for h in range(self.hidden.n_out):
            pred_reward += self.reward_weights[h] * hidden_rates[h]
        pred_energy = self.energy_bias
        for h in range(self.hidden.n_out):
            pred_energy += self.energy_weights[h] * hidden_rates[h]
        logits_cause = []
        for idx in range(self.cause_dim):
            logit = self.cause_bias[idx]
            for h in range(self.hidden.n_out):
                logit += self.cause_weights[h][idx] * hidden_rates[h]
            logits_cause.append(logit)
        probs_cause = softmax(logits_cause)
        energy_norm = sum(hidden_counts) / float(
            max(1, self.inner_steps * self.hidden.n_out)
        )
        return SelfModelState(
            probs_next=probs_next,
            probs_cause=probs_cause,
            pred_reward=pred_reward,
            pred_energy=pred_energy,
            eligibility_history=eligibility_history,
            bias_history=bias_history,
            hidden_rates=hidden_rates,
            hidden_counts=hidden_counts,
            energy_norm=energy_norm,
        )

    def update(
        self,
        state: SelfModelState,
        next_obs_index: int,
        reward_target: float,
        energy_target: float,
        cause_label: int,
    ) -> None:
        target_next = [0.0 for _ in range(self.next_obs_dim)]
        target_next[next_obs_index] = 1.0
        obs_errors = [
            self.obs_weight * (state.probs_next[idx] - target_next[idx])
            for idx in range(self.next_obs_dim)
        ]
        reward_error = self.reward_weight * (state.pred_reward - reward_target)
        energy_error = self.energy_weight * (state.pred_energy - energy_target)
        target_cause = [0.0, 0.0]
        target_cause[cause_label] = 1.0
        cause_errors = [
            self.cause_weight * (state.probs_cause[idx] - target_cause[idx])
            for idx in range(self.cause_dim)
        ]
        learning_signals = []
        for h in range(self.hidden.n_out):
            signal = 0.0
            for idx in range(self.next_obs_dim):
                signal += obs_errors[idx] * self.next_weights[h][idx]
            signal += reward_error * self.reward_weights[h]
            signal += energy_error * self.energy_weights[h]
            for idx in range(self.cause_dim):
                signal += cause_errors[idx] * self.cause_weights[h][idx]
            learning_signals.append(clip_value(signal, self.clip))
        for i in range(self.hidden.n_in):
            for h in range(self.hidden.n_out):
                grad = 0.0
                for elig in state.eligibility_history:
                    grad += learning_signals[h] * elig[i][h]
                grad = clip_value(grad, self.clip)
                self.hidden.weights[i][h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            grad = 0.0
            for bias_elig in state.bias_history:
                grad += learning_signals[h] * bias_elig[h]
            grad = clip_value(grad, self.clip)
            self.hidden.bias[h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            for idx in range(self.next_obs_dim):
                grad = clip_value(obs_errors[idx] * state.hidden_rates[h], self.clip)
                self.next_weights[h][idx] -= self.readout_lr * grad
        for idx in range(self.next_obs_dim):
            grad = clip_value(obs_errors[idx], self.clip)
            self.next_bias[idx] -= self.readout_lr * grad
        for h in range(self.hidden.n_out):
            grad = clip_value(reward_error * state.hidden_rates[h], self.clip)
            self.reward_weights[h] -= self.readout_lr * grad
        self.reward_bias -= self.readout_lr * clip_value(reward_error, self.clip)
        for h in range(self.hidden.n_out):
            grad = clip_value(energy_error * state.hidden_rates[h], self.clip)
            self.energy_weights[h] -= self.readout_lr * grad
        self.energy_bias -= self.readout_lr * clip_value(energy_error, self.clip)
        for h in range(self.hidden.n_out):
            for idx in range(self.cause_dim):
                grad = clip_value(cause_errors[idx] * state.hidden_rates[h], self.clip)
                self.cause_weights[h][idx] -= self.readout_lr * grad
        for idx in range(self.cause_dim):
            grad = clip_value(cause_errors[idx], self.clip)
            self.cause_bias[idx] -= self.readout_lr * grad


def build_self_model_input(
    state_index: int,
    action: int,
    mean_count: float,
    sum_count: float,
    eta_e: float,
    vth: float,
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
        max(0.0, min(vth, 1.0)),
    ]
    return obs_vec + action_vec + extras


def run_training(episodes: int = 90) -> None:
    setup_logging()
    logger = get_logger(__name__)
    random.seed(42)

    env = GridWorld(GridWorldConfig())
    policy_params = LIFParams(v_th=0.52, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    policy = SNNPolicy(
        state_size=env.cfg.size * env.cfg.size,
        hidden_size=24,
        params=policy_params,
    )
    self_params = LIFParams(v_th=0.5, tau_m=10.0, tau_a=20.0, beta=0.35, refractory=2)
    self_model = SelfModel(
        input_size=env.cfg.size * env.cfg.size + 4 + 4,
        hidden_size=28,
        params=self_params,
    )

    os.makedirs("runs", exist_ok=True)
    csv_path = os.path.join("runs", "self_model_metrics.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "episode",
                "steps",
                "avg_nll",
                "reward_mse",
                "energy_mse",
                "cause_acc",
            ],
        )
        writer.writeheader()

    visit_counts: Dict[int, int] = defaultdict(int)
    beta_bonus = 0.2
    episode_records: List[Tuple[float, float]] = []
    nll_history: List[float] = []
    cause_history: List[float] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        state_index = env.state_index(state)
        episode_reward = 0.0
        episode_nll = 0.0
        episode_reward_mse = 0.0
        episode_energy_mse = 0.0
        episode_cause_hits = 0
        steps = 0

        while steps < env.cfg.max_steps:
            policy_state = policy.forward(state_index)
            action = policy.sample_action(policy_state.probs)
            visit_counts[state_index] += 1
            bonus = beta_bonus / math.sqrt(visit_counts[state_index])
            features = build_self_model_input(
                state_index=state_index,
                action=action,
                mean_count=policy_state.mean_rate,
                sum_count=policy_state.sum_rate,
                eta_e=policy.hidden_lr,
                vth=policy_params.v_th,
                state_size=env.cfg.size * env.cfg.size,
            )
            self_state = self_model.forward(features)
            next_state, base_reward, done, caused_by_self = env.step(action)
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - policy.baseline
            policy.update(policy_state, action, advantage)
            policy.update_baseline(reward)
            next_index = env.state_index(next_state)

            energy_target = min(
                sum(policy_state.hidden_counts)
                / float(policy.inner_steps * policy.hidden.n_out),
                1.0,
            )
            self_model.update(
                state=self_state,
                next_obs_index=next_index,
                reward_target=reward,
                energy_target=energy_target,
                cause_label=1 if caused_by_self else 0,
            )

            nll = -math.log(max(self_state.probs_next[next_index], 1e-8))
            reward_mse = (self_state.pred_reward - reward) ** 2
            energy_mse = (self_state.pred_energy - energy_target) ** 2
            cause_pred = 1 if self_state.probs_cause[1] >= self_state.probs_cause[0] else 0
            cause_hit = 1 if cause_pred == (1 if caused_by_self else 0) else 0

            episode_nll += nll
            episode_reward_mse += reward_mse
            episode_energy_mse += energy_mse
            episode_cause_hits += cause_hit

            state = next_state
            state_index = next_index
            steps += 1
            if done:
                break

        avg_nll = episode_nll / max(steps, 1)
        avg_reward_mse = episode_reward_mse / max(steps, 1)
        avg_energy_mse = episode_energy_mse / max(steps, 1)
        cause_acc = episode_cause_hits / float(max(steps, 1))
        nll_history.append(avg_nll)
        cause_history.append(cause_acc)
        episode_records.append((episode_reward, cause_acc))

        with open(csv_path, "a", newline="") as f_csv:
            writer = csv.DictWriter(
                f_csv,
                fieldnames=[
                    "episode",
                    "steps",
                    "avg_nll",
                    "reward_mse",
                    "energy_mse",
                    "cause_acc",
                ],
            )
            writer.writerow(
                {
                    "episode": episode,
                    "steps": steps,
                    "avg_nll": avg_nll,
                    "reward_mse": avg_reward_mse,
                    "energy_mse": avg_energy_mse,
                    "cause_acc": cause_acc,
                }
            )

        logger.info(
            "回合 %03d 总回报 %.3f NLL %.3f cause_acc %.2f",
            episode,
            episode_reward,
            avg_nll,
            cause_acc,
        )

    if len(nll_history) >= 10:
        start_nll = sum(nll_history[:5]) / 5.0
        end_nll = sum(nll_history[-5:]) / 5.0
    else:
        start_nll = nll_history[0]
        end_nll = nll_history[-1]
    cause_tail = sum(cause_history[-min(20, len(cause_history)) :]) / float(
        min(20, len(cause_history))
    )
    logger.info(
        "起始 NLL %.3f 末尾 NLL %.3f cause_acc(尾部) %.2f",
        start_nll,
        end_nll,
        cause_tail,
    )
    assert end_nll < start_nll, "Self-Model NLL 未下降。"
    assert cause_tail > 0.7, "自因分类准确率未超过 0.7。"


if __name__ == "__main__":
    run_training()
