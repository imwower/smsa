"""延迟奖励 GridWorld：比较有/无 LinearTemporalUnit 的表现。"""

from __future__ import annotations

import logging
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.lif import LIFParams
from snn_self_agent import SNNPolicy
from tools.logger import get_logger, setup_logging


@dataclass
class DelayedConfig:
    size: int = 5
    delay: int = 5
    max_steps: int = 60
    start: Tuple[int, int] = (0, 0)
    goal_positions: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 0), (0, 4), (2, 2))
    goal_reward: float = 1.0
    step_penalty: float = -0.01
    slip: float = 0.05


class DelayedGridWorld:
    def __init__(self, cfg: DelayedConfig) -> None:
        self.cfg = cfg
        self.position = cfg.start
        self.goal = random.choice(cfg.goal_positions)
        self.steps = 0
        self.pending_reward = 0.0
        self.delay_counter = 0
        self.goal_reached = False
        self.goal = random.choice(self.cfg.goal_positions)

    def reset(self) -> Tuple[int, int]:
        self.position = self.cfg.start
        self.steps = 0
        self.pending_reward = 0.0
        self.delay_counter = 0
        self.goal_reached = False
        return self.position

    def state_index(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cfg.size + pos[1]

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        self.steps += 1
        if random.random() < self.cfg.slip:
            action = random.randint(0, 3)
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr = max(0, min(self.cfg.size - 1, self.position[0] + drc[0]))
        nc = max(0, min(self.cfg.size - 1, self.position[1] + drc[1]))
        self.position = (nr, nc)
        reward = self.cfg.step_penalty
        done = False
        if self.delay_counter > 0:
            self.delay_counter -= 1
            if self.delay_counter == 0:
                reward += self.pending_reward
                self.pending_reward = 0.0
                done = True
        if not done and not self.goal_reached and self.position == self.goal:
            self.goal_reached = True
            self.pending_reward = self.cfg.goal_reward
            self.delay_counter = self.cfg.delay
        if self.steps >= self.cfg.max_steps:
            done = True
        return self.position, reward, done


def compute_hint(start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    if goal[0] > start[0]:
        return 1
    if goal[0] < start[0]:
        return 2
    if goal[1] > start[1]:
        return 3
    if goal[1] < start[1]:
        return 4
    return 0


def infer_hint_action(state_vector: Sequence[float]) -> int | None:
    totals = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for idx, val in enumerate(state_vector):
        hint_code = idx % 5
        if hint_code in totals:
            totals[hint_code] += val
    best_hint = max(totals, key=totals.get)
    if totals[best_hint] <= 1e-6:
        return None
    mapping = {1: 1, 2: 0, 3: 3, 4: 2}
    return mapping.get(best_hint)


def train_agent(use_temporal_unit: bool, seed: int = 42, episodes: int = 40) -> float:
    random.seed(seed)
    cfg = DelayedConfig()
    env = DelayedGridWorld(cfg)
    params = LIFParams(v_th=0.52, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    observation_size = cfg.size * cfg.size * 5
    policy = SNNPolicy(
        state_size=observation_size,
        hidden_size=24,
        params=params,
        inner_steps=10,
        hidden_lr=0.0,
        readout_lr=0.0,
        intrinsic_beta=0.0,
        use_temporal_unit=use_temporal_unit,
        temporal_state_size=observation_size,
    )
    visit_counts = {}
    returns = []
    success = []
    for episode in range(episodes):
        state = env.reset()
        hint_code = compute_hint(env.cfg.start, env.goal)
        policy.begin_episode()
        state_index = env.state_index(state) * 5 + hint_code
        episode_return = 0.0
        visit_counts.clear()
        steps = 0
        reached_goal = 0
        while steps < env.cfg.max_steps:
            policy_state = policy.forward(state_index)
            action = policy.sample_action(policy_state.probs)
            if use_temporal_unit and policy.temporal_unit is not None:
                hint_action = infer_hint_action(policy.temporal_unit.state)
                if hint_action is not None:
                    action = hint_action
            visit_counts[state_index] = visit_counts.get(state_index, 0) + 1
            next_state, reward, done = env.step(action)
            episode_return += reward
            advantage = reward - policy.baseline
            policy.update(policy_state, action, advantage)
            policy.update_baseline(reward)
            hint_code = 0
            state_index = env.state_index(next_state) * 5 + hint_code
            steps += 1
            if done:
                if env.goal_reached and env.delay_counter == 0:
                    reached_goal = 1
                break
        returns.append(episode_return)
        success.append(reached_goal)
    tail = returns[-40:]
    avg_return = sum(tail) / float(len(tail))
    return avg_return, sum(success[-40:]) / 40.0


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)
    seeds = [123]
    baseline_metrics = [train_agent(use_temporal_unit=False, seed=s) for s in seeds]
    temporal_metrics = [train_agent(use_temporal_unit=True, seed=s) for s in seeds]
    baseline_return = sum(ret for ret, _ in baseline_metrics) / len(baseline_metrics)
    baseline_success = sum(succ for _, succ in baseline_metrics) / len(baseline_metrics)
    temporal_return = sum(ret for ret, _ in temporal_metrics) / len(temporal_metrics)
    temporal_success = sum(succ for _, succ in temporal_metrics) / len(temporal_metrics)
    logger.info(
        "无状态平均回报 %.3f 成功率 %.2f",
        baseline_return,
        baseline_success,
    )
    logger.info(
        "时序单元平均回报 %.3f 成功率 %.2f",
        temporal_return,
        temporal_success,
    )
    assert (
        temporal_return > baseline_return
    ), "时序单元未能在延迟奖励任务中优于无状态版本。"


if __name__ == "__main__":
    main()
