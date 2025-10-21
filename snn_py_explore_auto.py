"""GridWorld 主动探索 + REINFORCE + 内在动机的单文件原型。"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate
from tools.logger import get_logger, setup_logging


# ----------------------------- 环境定义 ------------------------------ #


Action = int  # 0:上, 1:下, 2:左, 3:右


@dataclass
class GridWorldConfig:
    size: int = 5
    slip: float = 0.1
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

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        self.steps += 1
        if random.random() < self.cfg.slip:
            action = random.randint(0, 3)
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
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
        return self.position, reward, done


# ----------------------------- SNN 策略 ------------------------------ #


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def poisson_spikes(bits: Sequence[int], high_rate: float, low_rate: float) -> List[int]:
    spikes: List[int] = []
    for bit in bits:
        rate = high_rate if bit else low_rate
        spikes.append(1 if random.random() < rate else 0)
    return spikes


def clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class SNNAgent:
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

    def forward(
        self,
        state_index: int,
    ) -> Tuple[List[float], List[List[List[float]]], List[List[float]], List[float]]:
        bits = self.encode_state(state_index)
        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]
        eligibility_history: List[List[List[float]]] = []
        bias_history: List[List[float]] = []
        for _ in range(self.inner_steps):
            pre_spikes = poisson_spikes(bits, self.high_rate, self.low_rate)
            spikes, _, eligibility_snapshot, bias_snapshot = self.hidden.step(
                pre_spikes
            )
            hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]
            eligibility_history.append(copy.deepcopy(eligibility_snapshot))
            bias_history.append(bias_snapshot[:])
        hidden_rates = [count / float(self.inner_steps) for count in hidden_counts]
        logits = []
        for action in range(4):
            logit = self.readout_bias[action]
            for h in range(self.hidden.n_out):
                logit += self.readout_weights[h][action] * hidden_rates[h]
            logits.append(logit)
        probs = softmax(logits)
        return probs, eligibility_history, bias_history, hidden_rates

    def sample_action(
        self,
        probs: Sequence[float],
    ) -> int:
        threshold = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                return idx
        return len(probs) - 1

    def update_baseline(self, reward: float) -> None:
        self.baseline = (1.0 - self.baseline_beta) * self.baseline + self.baseline_beta * reward

    def update(
        self,
        probs: Sequence[float],
        action: int,
        hidden_rates: Sequence[float],
        eligibility_history: Sequence[List[List[float]]],
        bias_history: Sequence[List[float]],
        advantage: float,
    ) -> None:
        policy_error = [p for p in probs]
        policy_error[action] -= 1.0
        policy_error = [clip_value(err * advantage, self.clip) for err in policy_error]
        learning_signals = []
        for h in range(self.hidden.n_out):
            signal = 0.0
            for a in range(4):
                signal += policy_error[a] * self.readout_weights[h][a]
            learning_signals.append(clip_value(signal, self.clip))
        for i in range(self.hidden.n_in):
            for h in range(self.hidden.n_out):
                grad = 0.0
                for elig in eligibility_history:
                    grad += learning_signals[h] * elig[i][h]
                grad = clip_value(grad, self.clip)
                self.hidden.weights[i][h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            grad = 0.0
            for bias_elig in bias_history:
                grad += learning_signals[h] * bias_elig[h]
            grad = clip_value(grad, self.clip)
            self.hidden.bias[h] -= self.hidden_lr * grad
        for h in range(self.hidden.n_out):
            for a in range(4):
                grad = clip_value(policy_error[a] * hidden_rates[h], self.clip)
                self.readout_weights[h][a] -= self.readout_lr * grad
        for a in range(4):
            grad = clip_value(policy_error[a], self.clip)
            self.readout_bias[a] -= self.readout_lr * grad


# ----------------------------- 训练循环 ------------------------------ #


def run_training(episodes: int = 240) -> None:
    setup_logging()
    logger = get_logger(__name__)

    env = GridWorld(GridWorldConfig())
    params = LIFParams(v_th=0.5, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    agent = SNNAgent(
        state_size=env.cfg.size * env.cfg.size,
        hidden_size=24,
        params=params,
    )
    visit_counts: Dict[int, int] = defaultdict(int)
    beta_bonus = 0.35
    running_return = None
    success_history: List[int] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        state_index = env.state_index(state)
        episode_reward = 0.0
        reached_goal = 0
        for _ in range(env.cfg.max_steps):
            probs, elig_history, bias_history, hidden_rates = agent.forward(state_index)
            action = agent.sample_action(probs)
            next_state, base_reward, done = env.step(action)
            visit_key = env.state_index(state)
            bonus = beta_bonus / math.sqrt(visit_counts[visit_key] + 1)
            visit_counts[visit_key] += 1
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - agent.baseline
            agent.update(
                probs=probs,
                action=action,
                hidden_rates=hidden_rates,
                eligibility_history=elig_history,
                bias_history=bias_history,
                advantage=advantage,
            )
            agent.update_baseline(reward)
            state = next_state
            state_index = env.state_index(state)
            if done:
                if state == env.cfg.goal:
                    reached_goal = 1
                break
        success_history.append(reached_goal)
        if running_return is None:
            running_return = episode_reward
        else:
            smoothed = 0.9 * running_return + 0.1 * episode_reward
            running_return = max(running_return, smoothed)
        window = success_history[-40:]
        success_rate = sum(window) / float(len(window))
        logger.info(
            "回合 %03d 平均回报 %.3f 成功率 %.2f",
            episode,
            running_return,
            success_rate,
        )

    overall_success = sum(success_history[-60:]) / float(min(60, len(success_history)))
    logger.info("最近 60 回合成功率 %.2f", overall_success)
    assert overall_success > 0.6, "终点成功率未超过 60%。"


if __name__ == "__main__":
    run_training()
