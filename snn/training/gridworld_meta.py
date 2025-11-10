"""Training loop for GridWorld exploration with meta adaptation."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence

from envs.gridworld import GridWorld
from meta.autoadapt import MetaLearner
from snn.agents.gridworld_agent import GridworldAgent, build_self_model_input
from snn.lif import LIFParams
from snn.selfmodel import SelfModel
from tools.logger import get_logger


logger = get_logger(__name__)


@dataclass
class GridWorldConfig:
    """Environment parameters mirrored from the single-file prototype."""

    size: int = 5
    slip_prob: float = 0.05
    max_steps: int = 60
    step_cost: float = -0.01
    goal_reward: float = 1.0

    def make_env(self, *, seed: int | None = None) -> GridWorld:
        return GridWorld(
            size=self.size,
            slip_prob=self.slip_prob,
            step_cost=self.step_cost,
            goal_reward=self.goal_reward,
            max_steps=self.max_steps,
            seed=seed,
        )


def _obs_to_index(obs: Sequence[float]) -> int:
    return max(range(len(obs)), key=lambda idx: obs[idx])


def evaluate_agent(
    agent: GridworldAgent,
    env_cfg: GridWorldConfig,
    *,
    episodes: int,
    seed: int,
) -> float:
    state = random.getstate()
    random.seed(seed)
    total_reward = 0.0
    for episode in range(episodes):
        env_seed = seed * 997 + episode * 131
        env = env_cfg.make_env(seed=env_seed)
        visit_counts: Dict[int, int] = defaultdict(int)
        obs = env.reset()
        idx = _obs_to_index(obs)
        agent.begin_episode()
        steps = 0
        while steps < env_cfg.max_steps:
            pol_state = agent.forward(idx, collect_traces=False)
            action = agent.sample_action(pol_state.probs)
            bonus = agent.intrinsic_bonus(visit_counts[idx])
            visit_counts[idx] += 1
            obs, base_reward, done, _info = env.step(action)
            reward = base_reward + bonus
            total_reward += reward
            idx = _obs_to_index(obs)
            steps += 1
            if done:
                break
    random.setstate(state)
    return total_reward / float(max(episodes, 1))


def train_gridworld(
    *,
    episodes: int = 240,
    seed: int | None = None,
    env_cfg: GridWorldConfig | None = None,
    validate: bool = True,
) -> Dict[str, float]:
    if seed is not None:
        random.seed(seed)

    env_cfg = env_cfg or GridWorldConfig()
    env = env_cfg.make_env(seed=random.randrange(1_000_000))
    state_size = env_cfg.size * env_cfg.size

    params = LIFParams(v_th=0.5, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    agent = GridworldAgent(
        state_size=state_size,
        hidden_size=24,
        params=params,
    )
    self_params = LIFParams(v_th=0.5, tau_m=10.0, tau_a=20.0, beta=0.35, refractory=2)
    self_model = SelfModel(
        obs_dim=state_size,
        action_dim=4,
        hidden_size=24,
        lif_params=self_params,
    )
    visit_counts: Dict[int, int] = defaultdict(int)
    meta = MetaLearner(window=20, min_delta=0.05)
    return_history: deque[float] = deque(maxlen=meta.window)
    running_return: float | None = None
    success_history: List[int] = []
    nll_history: List[float] = []
    cause_history: List[float] = []
    energy_history: List[float] = []
    policy_alpha = 1.0
    policy_beta = 0.4
    meta_effect_span = 12
    meta_recent_steps = 0

    for episode in range(1, episodes + 1):
        obs = env.reset()
        agent.begin_episode()
        state_index = _obs_to_index(obs)
        episode_reward = 0.0
        reached_goal = 0
        episode_nll = 0.0
        episode_cause_hits = 0
        episode_energy_mse = 0.0
        steps = 0
        while steps < env_cfg.max_steps:
            policy_state = agent.forward(state_index)
            action = agent.sample_action(policy_state.probs)
            bonus = agent.intrinsic_bonus(visit_counts[state_index])
            visit_counts[state_index] += 1
            features = build_self_model_input(
                state_index=state_index,
                action=action,
                mean_rate=policy_state.mean_rate,
                sum_rate=policy_state.sum_rate,
                eta_e=agent.hidden_lr,
                v_th=agent.hidden.params.v_th,
                state_size=state_size,
            )
            self_state = self_model.forward(features)

            obs, base_reward, done, info = env.step(action)
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - agent.baseline
            next_index = _obs_to_index(obs)

            energy_target = min(
                sum(policy_state.hidden_counts)
                / float(agent.inner_steps * agent.hidden.n_out),
                1.0,
            )
            cause_label = 1 if meta_recent_steps > 0 else 0
            self_signal = self_model.update(
                state=self_state,
                next_obs_index=next_index,
                reward_target=reward,
                energy_target=energy_target,
                cause_label=cause_label,
            )
            agent.update(
                policy_state,
                action,
                advantage,
                self_signal=self_signal,
                alpha=policy_alpha,
                beta=policy_beta,
            )
            agent.update_baseline(reward)
            if meta_recent_steps > 0:
                meta_recent_steps -= 1

            nll = -math.log(max(self_state.probs_next[next_index], 1e-8))
            cause_pred = 1 if self_state.probs_cause[1] >= self_state.probs_cause[0] else 0
            cause_hit = 1 if cause_pred == cause_label else 0
            energy_mse = (self_state.pred_energy - energy_target) ** 2

            episode_nll += nll
            episode_cause_hits += cause_hit
            episode_energy_mse += energy_mse

            state_index = next_index
            steps += 1
            if done:
                if info.get("goal_reached", False):
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
        avg_nll = episode_nll / float(max(steps, 1))
        avg_cause = episode_cause_hits / float(max(steps, 1))
        avg_energy = episode_energy_mse / float(max(steps, 1))
        nll_history.append(avg_nll)
        cause_history.append(avg_cause)
        energy_history.append(avg_energy)
        logger.info(
            "回合 %03d 平均回报 %.3f 成功率 %.2f NLL %.3f cause_acc %.2f",
            episode,
            running_return,
            success_rate,
            avg_nll,
            avg_cause,
        )
        return_history.append(running_return)
        if meta.should_trigger(list(return_history)):

            def evaluate_for_meta(candidate_agent: GridworldAgent, seed: int) -> float:
                return evaluate_agent(
                    candidate_agent,
                    env_cfg,
                    episodes=meta.ab_episodes,
                    seed=seed,
                )

            agent, meta_messages = meta.adapt(
                agent,
                step=episode,
                evaluate_fn=evaluate_for_meta,
            )
            for meta_msg in meta_messages:
                logger.info(meta_msg)
            if any("accepted" in msg for msg in meta_messages):
                meta_recent_steps = meta_effect_span
            return_history.clear()

    overall_success = sum(success_history[-60:]) / float(min(60, len(success_history)))
    logger.info("最近 60 回合成功率 %.2f", overall_success)
    if nll_history:
        window = min(20, len(nll_history))
        tail_nll = sum(nll_history[-window:]) / float(window)
        tail_cause = sum(cause_history[-window:]) / float(window)
        tail_energy = sum(energy_history[-window:]) / float(window)
        logger.info(
            "尾部指标 NLL %.3f cause_acc %.2f energy_mse %.3f",
            tail_nll,
            tail_cause,
            tail_energy,
        )
    else:
        tail_nll = tail_cause = tail_energy = 0.0
    if validate:
        assert overall_success > 0.6, "终点成功率未超过 60%。"
        if cause_history:
            assert (
                sum(cause_history[-20:]) / float(min(20, len(cause_history)))
                >= 0.7
            ), "自因分类准确率未达到 0.7。"
        assert meta.positive_ratio() >= 0.5, "自改 Δ>0 的比例未达到 50%。"
    return {
        "overall_success": overall_success,
        "tail_nll": tail_nll,
        "tail_cause": tail_cause,
        "tail_energy": tail_energy,
        "positive_ratio": meta.positive_ratio(),
    }


__all__ = ["GridWorldConfig", "train_gridworld", "evaluate_agent"]
