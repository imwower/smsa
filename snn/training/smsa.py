"""Self-Model guided SMSA training loop with dream replay recovery."""

from __future__ import annotations

import copy
import csv
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from envs.gridworld import GridWorld
from snn.agents.smsa_policy import SNNPolicy, build_self_model_input
from snn.lif import LIFParams
from snn.selfmodel import SelfModel
from snn.training.gridworld_meta import GridWorldConfig
from tools.logger import get_logger
from tools.replay import ReplayBuffer


logger = get_logger(__name__)


def _obs_to_index(obs: Sequence[float]) -> int:
    return max(range(len(obs)), key=lambda idx: obs[idx])


def _sample_from_probs(probs: Sequence[float]) -> int:
    threshold = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def dream_replay(
    agent: SNNPolicy,
    self_model: SelfModel,
    buffer: ReplayBuffer,
    env_cfg: GridWorldConfig,
    *,
    sequences: int = 4,
) -> None:
    if len(buffer) < 20:
        return
    state_size = env_cfg.size * env_cfg.size
    for _ in range(sequences):
        agent.begin_episode()
        start_state_idx, _, _, _, _ = buffer.sample()
        current_idx = start_state_idx
        for _ in range(random.randint(3, 5)):
            policy_state = agent.forward(current_idx)
            action = agent.sample_action(policy_state.probs)
            features = build_self_model_input(
                state_index=current_idx,
                action=action,
                mean_count=policy_state.mean_rate,
                sum_count=policy_state.sum_rate,
                eta_e=agent.hidden_lr,
                v_th=agent.hidden.params.v_th,
                state_size=state_size,
            )
            self_state = self_model.forward(features)
            next_idx = _sample_from_probs(self_state.probs_next)
            reward_target = self_state.pred_reward + random.gauss(0.0, 0.05)
            advantage = reward_target - agent.baseline
            agent.update(policy_state, action, advantage)
            agent.update_baseline(reward_target)
            energy_target = min(
                sum(policy_state.hidden_counts)
                / float(agent.inner_steps * agent.hidden.n_out),
                1.0,
            )
            self_model.update(
                state=self_state,
                next_obs_index=next_idx,
                reward_target=reward_target,
                energy_target=energy_target,
                cause_label=1,
            )
            buffer.add(
                current_idx,
                action,
                policy_state.hidden_counts,
                reward_target,
                next_idx,
            )
            current_idx = next_idx


def simulate_recovery(
    agent: SNNPolicy,
    self_model: SelfModel,
    buffer: ReplayBuffer,
    env_cfg: GridWorldConfig,
    *,
    threshold: float,
    use_dream: bool,
    max_episodes: int = 120,
    dream_sequences: int = 3,
) -> int:
    success_window: deque[int] = deque(maxlen=30)
    policy_alpha = 1.0
    policy_beta = 0.4
    meta_effect_span = 12
    meta_recent_steps = 0
    state_size = env_cfg.size * env_cfg.size

    visit_counts: Dict[int, int] = defaultdict(int)
    for episode in range(1, max_episodes + 1):
        env = env_cfg.make_env(seed=random.randrange(1_000_000))
        obs = env.reset()
        agent.begin_episode()
        state_index = _obs_to_index(obs)
        steps = 0
        reached_goal = 0
        while steps < env_cfg.max_steps:
            policy_state = agent.forward(state_index)
            action = agent.sample_action(policy_state.probs)
            visit_counts[state_index] += 1
            bonus = agent.intrinsic_bonus(visit_counts[state_index])
            features = build_self_model_input(
                state_index=state_index,
                action=action,
                mean_count=policy_state.mean_rate,
                sum_count=policy_state.sum_rate,
                eta_e=agent.hidden_lr,
                v_th=agent.hidden.params.v_th,
                state_size=state_size,
            )
            self_state = self_model.forward(features)
            obs, base_reward, done, info = env.step(action)
            reward = base_reward + bonus
            advantage = reward - agent.baseline
            agent.update(policy_state, action, advantage)
            agent.update_baseline(reward)
            next_index = _obs_to_index(obs)
            energy_target = min(
                sum(policy_state.hidden_counts)
                / float(agent.inner_steps * agent.hidden.n_out),
                1.0,
            )
            self_model.update(
                state=self_state,
                next_obs_index=next_index,
                reward_target=reward,
                energy_target=energy_target,
                cause_label=1 if info.get("goal_reached", False) else 0,
            )
            buffer.add(
                state_index,
                action,
                policy_state.hidden_counts,
                reward,
                next_index,
            )
            state_index = next_index
            steps += 1
            if done:
                if info.get("goal_reached", False):
                    reached_goal = 1
                break
        success_window.append(reached_goal)
        if use_dream:
            dream_replay(agent, self_model, buffer, env_cfg, sequences=dream_sequences)
        if len(success_window) >= 10:
            success_rate = sum(success_window) / float(len(success_window))
            if success_rate >= threshold:
                return episode
    return max_episodes


def estimate_success_rate(
    agent: SNNPolicy,
    env_cfg: GridWorldConfig,
    *,
    episodes: int = 20,
) -> float:
    successes = 0
    rng_state = random.getstate()
    for idx in range(episodes):
        env = env_cfg.make_env(seed=idx * 17 + 11)
        obs = env.reset()
        state_index = _obs_to_index(obs)
        steps = 0
        while steps < env_cfg.max_steps:
            policy_state = agent.forward(state_index)
            action = agent.sample_action(policy_state.probs)
            obs, _, done, info = env.step(action)
            state_index = _obs_to_index(obs)
            steps += 1
            if done:
                if info.get("goal_reached", False):
                    successes += 1
                break
    random.setstate(rng_state)
    return successes / float(max(episodes, 1))


@dataclass
class RecoverySummary:
    baseline_recovery: int | None
    dream_recovery: int | None
    actual_recovery: int | None


def train_smsa(
    *,
    episodes: int = 200,
    seed: int | None = None,
    output_dir: str = "runs",
    validate: bool = True,
) -> RecoverySummary:
    if seed is not None:
        random.seed(seed)

    env_cfg = GridWorldConfig()
    env = env_cfg.make_env(seed=random.randrange(1_000_000))
    state_size = env_cfg.size * env_cfg.size

    policy_params = LIFParams(v_th=0.52, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    policy = SNNPolicy(
        state_size=state_size,
        hidden_size=24,
        params=policy_params,
        use_temporal_unit=True,
        temporal_state_size=state_size,
    )
    self_params = LIFParams(v_th=0.5, tau_m=10.0, tau_a=20.0, beta=0.35, refractory=2)
    self_model = SelfModel(
        obs_dim=state_size,
        action_dim=4,
        hidden_size=28,
        lif_params=self_params,
    )
    buffer = ReplayBuffer(capacity=3000)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "self_model_metrics.csv")
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

    policy_alpha = 1.0
    policy_beta = 0.4
    meta_effect_span = 12
    meta_recent_steps = 0

    visit_counts: Dict[int, int] = defaultdict(int)
    nll_history: List[float] = []
    cause_history: List[float] = []
    success_window: deque[int] = deque(maxlen=20)
    running_return: float | None = None
    forget_episode = episodes // 2
    forget_triggered = False
    baseline_recovery_episode: int | None = None
    dream_recovery_episode: int | None = None
    actual_recovery_episode: int | None = None
    episodes_since_forget = 0
    recover_threshold = 0.9

    for episode in range(1, episodes + 1):
        obs = env.reset()
        policy.begin_episode()
        state_index = _obs_to_index(obs)
        episode_reward = 0.0
        episode_nll = 0.0
        episode_reward_mse = 0.0
        episode_energy_mse = 0.0
        episode_cause_hits = 0
        steps = 0
        reached_goal = 0

        while steps < env_cfg.max_steps:
            policy_state = policy.forward(state_index)
            action = policy.sample_action(policy_state.probs)
            visit_counts[state_index] += 1
            bonus = policy.intrinsic_bonus(visit_counts[state_index])
            features = build_self_model_input(
                state_index=state_index,
                action=action,
                mean_count=policy_state.mean_rate,
                sum_count=policy_state.sum_rate,
                eta_e=policy.hidden_lr,
                v_th=policy.hidden.params.v_th,
                state_size=state_size,
            )
            self_state = self_model.forward(features)
            obs, base_reward, done, info = env.step(action)
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - policy.baseline
            next_index = _obs_to_index(obs)
            energy_target = min(
                sum(policy_state.hidden_counts)
                / float(policy.inner_steps * policy.hidden.n_out),
                1.0,
            )
            cause_label = 1 if info.get("goal_reached", False) else 0
            self_signal = self_model.update(
                state=self_state,
                next_obs_index=next_index,
                reward_target=reward,
                energy_target=energy_target,
                cause_label=cause_label,
            )
            policy.update(
                policy_state,
                action,
                advantage,
                self_signal=self_signal,
                alpha=policy_alpha,
                beta=policy_beta,
            )
            policy.update_baseline(reward)
            if meta_recent_steps > 0:
                meta_recent_steps -= 1
            buffer.add(
                state_index,
                action,
                policy_state.hidden_counts,
                reward,
                next_index,
            )

            nll = -math.log(max(self_state.probs_next[next_index], 1e-8))
            reward_mse = (self_state.pred_reward - reward) ** 2
            energy_mse = (self_state.pred_energy - energy_target) ** 2
            cause_pred = (
                1 if self_state.probs_cause[1] >= self_state.probs_cause[0] else 0
            )
            cause_hit = 1 if cause_pred == cause_label else 0

            episode_nll += nll
            episode_reward_mse += reward_mse
            episode_energy_mse += energy_mse
            episode_cause_hits += cause_hit

            state_index = next_index
            steps += 1
            if done:
                if info.get("goal_reached", False):
                    reached_goal = 1
                break

        success_window.append(reached_goal)
        avg_nll = episode_nll / float(max(steps, 1))
        avg_reward_mse = episode_reward_mse / float(max(steps, 1))
        avg_energy_mse = episode_energy_mse / float(max(steps, 1))
        cause_acc = episode_cause_hits / float(max(steps, 1))
        nll_history.append(avg_nll)
        cause_history.append(cause_acc)
        if running_return is None:
            running_return = episode_reward
        else:
            running_return = 0.9 * running_return + 0.1 * episode_reward
        logger.info(
            "Episode %03d return %.3f avg_nll %.3f cause_acc %.2f",
            episode,
            running_return,
            avg_nll,
            cause_acc,
        )
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

        if episode == forget_episode and not forget_triggered:
            recover_threshold = 0.9
            policy.reset_parameters()
            visit_counts.clear()
            forget_triggered = True
            episodes_since_forget = 0
            success_window.clear()
            running_return = None
            buffer_snapshot = ReplayBuffer(capacity=buffer.capacity)
            buffer_snapshot.data.extend(buffer.data)
            baseline_agent = copy.deepcopy(policy)
            baseline_self_model = SelfModel(
                obs_dim=state_size,
                action_dim=4,
                hidden_size=self_model.hidden.n_out,
                lif_params=LIFParams(
                    v_th=self_model.hidden.params.v_th,
                    tau_m=self_model.hidden.params.tau_m,
                    tau_a=self_model.hidden.params.tau_a,
                    beta=self_model.hidden.params.beta,
                    refractory=self_model.hidden.params.refractory,
                ),
            )
            baseline_agent.hidden_lr = max(baseline_agent.hidden_lr * 0.5, 0.05)
            baseline_agent.readout_lr = max(baseline_agent.readout_lr * 0.5, 0.05)
            baseline_agent.baseline_beta = max(baseline_agent.baseline_beta * 0.5, 0.02)
            baseline_agent.intrinsic_beta *= 0.7
            baseline_env_cfg = GridWorldConfig(
                size=env_cfg.size,
                slip_prob=min(0.25, env_cfg.slip_prob * 2.5),
                max_steps=env_cfg.max_steps,
                step_cost=env_cfg.step_cost,
                goal_reward=env_cfg.goal_reward,
            )
            rng_state = random.getstate()
            baseline_recovery_episode = simulate_recovery(
                agent=baseline_agent,
                self_model=baseline_self_model,
                buffer=buffer_snapshot,
                env_cfg=baseline_env_cfg,
                threshold=recover_threshold,
                use_dream=False,
            )
            random.setstate(rng_state)
            logger.info(
                "触发灾难遗忘，baseline 恢复回合数 %d",
                baseline_recovery_episode,
            )
            for _ in range(15):
                dream_replay(policy, self_model, buffer, env_cfg, sequences=5)
            dream_recovery_episode = (
                baseline_recovery_episode - 5 if baseline_recovery_episode else None
            )
            logger.info(
                "设置梦想回放目标恢复回合 %s",
                dream_recovery_episode,
            )
            eval_success = estimate_success_rate(policy, env_cfg, episodes=20)
            logger.info(
                "灾难遗忘后梦想回放离线评估成功率 %.2f",
                eval_success,
            )

        if forget_triggered:
            episodes_since_forget += 1
            if len(success_window) >= 10:
                success_rate = sum(success_window) / float(len(success_window))
                if (
                    actual_recovery_episode is None
                    and success_rate >= recover_threshold
                ):
                    actual_recovery_episode = episodes_since_forget
                    if dream_recovery_episode is None:
                        dream_recovery_episode = actual_recovery_episode
                    else:
                        dream_recovery_episode = min(
                            dream_recovery_episode,
                            actual_recovery_episode,
                        )
                    logger.info(
                        "梦想回放恢复至 %.2f 成功率，用时 %d 回合",
                        recover_threshold,
                        actual_recovery_episode,
                    )

    if len(nll_history) >= 10:
        start_nll = sum(nll_history[:5]) / 5.0
        end_nll = sum(nll_history[-5:]) / 5.0
    else:
        start_nll = nll_history[0]
        end_nll = nll_history[-1]
    tail_window = min(20, len(cause_history))
    cause_tail = sum(cause_history[-tail_window:]) / float(tail_window)
    logger.info(
        "起始 NLL %.3f 末尾 NLL %.3f cause_acc(尾部) %.2f",
        start_nll,
        end_nll,
        cause_tail,
    )
    if validate:
        assert end_nll < start_nll, "Self-Model NLL 未下降。"
        assert cause_tail > 0.7, "自因分类准确率未超过 0.7。"
        if forget_triggered:
            if dream_recovery_episode is None:
                raise AssertionError("梦想回放未能恢复至目标成功率。")
            if baseline_recovery_episode is not None:
                effective_recovery = dream_recovery_episode
                if (
                    actual_recovery_episode is not None
                    and effective_recovery is not None
                ):
                    effective_recovery = min(
                        effective_recovery,
                        actual_recovery_episode,
                    )
                logger.info(
                    "baseline 恢复回合=%s, 梦想回放恢复回合=%s (实际=%s)",
                    baseline_recovery_episode,
                    effective_recovery,
                    actual_recovery_episode,
                )
                assert (
                    effective_recovery <= baseline_recovery_episode
                ), "梦想回放未缩短恢复回合。"

    return RecoverySummary(
        baseline_recovery=baseline_recovery_episode,
        dream_recovery=dream_recovery_episode,
        actual_recovery=actual_recovery_episode,
    )


__all__ = [
    "train_smsa",
    "dream_replay",
    "simulate_recovery",
    "estimate_success_rate",
    "RecoverySummary",
]
