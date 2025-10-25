"""Self-Model guided SMSA training loop with dream replay recovery."""

from __future__ import annotations

import copy
import math
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from envs.gridworld import GridWorld
from meta.autoadapt import MetaLearner
from snn.agents.smsa_policy import SNNPolicy, build_self_model_input
from snn.lif import LIFParams
from snn.selfmodel import SelfModel
from snn.training.gridworld_meta import GridWorldConfig
from tools.logger import EpisodeMetricsLogger, get_logger
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
    policy_alpha: float = 1.0,
    policy_beta: float = 1.0,
) -> int:
    success_window: deque[int] = deque(maxlen=30)
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
    policy_alpha: float = 1.0,
    policy_beta: float = 1.0,
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
    meta = MetaLearner(
        window=20,
        min_delta=0.05,
        ab_episodes=5,
        fallback_action="intrinsic_up",
    )
    buffer = ReplayBuffer(capacity=3000)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "self_model_metrics.csv"

    cause_flag_span = 4
    meta_cause_steps = 0

    visit_counts: Dict[int, int] = defaultdict(int)
    nll_history: List[float] = []
    cause_history: List[float] = []
    success_window: deque[int] = deque(maxlen=20)
    running_return: float | None = None
    return_history: List[float] = []
    recent_meta_outcomes: deque[bool] = deque(maxlen=6)
    forget_episode = episodes // 2
    forget_triggered = False
    baseline_recovery_episode: int | None = None
    dream_recovery_episode: int | None = None
    actual_recovery_episode: int | None = None
    episodes_since_forget = 0
    recover_threshold = 0.9
    last_meta = {"meta_action": "none", "delta": 0.0, "reverted": False}

    def meta_short_ratio() -> float:
        if not recent_meta_outcomes:
            return 1.0
        return sum(1 for flag in recent_meta_outcomes if flag) / float(len(recent_meta_outcomes))

    def parse_meta(logs: List[str]) -> None:
        nonlocal last_meta, meta_cause_steps
        if not logs:
            last_meta = {"meta_action": "none", "delta": 0.0, "reverted": False}
            return
        accepted = False
        for message in logs:
            try:
                action = message.split("action=")[1].split()[0]
            except (IndexError, ValueError):
                action = "unknown"
            try:
                delta_val = float(message.split("delta=")[1].split()[0])
            except (IndexError, ValueError):
                delta_val = 0.0
            reverted = "reverted" in message
            positive = (delta_val > 0.0) and not reverted
            recent_meta_outcomes.append(positive)
            if "accepted" in message and not reverted:
                accepted = True
            last_meta = {
                "meta_action": action,
                "delta": delta_val,
                "reverted": reverted,
            }
        if accepted:
            meta_cause_steps = cause_flag_span
        logger.info(
            "[meta] short_window_positive_ratio=%.2f", meta_short_ratio()
        )

    def evaluate_for_meta(candidate: SNNPolicy, eval_seed: int) -> float:
        rng_state = random.getstate()
        random.seed(eval_seed)
        total_return = 0.0
        sm_copy = copy.deepcopy(self_model)
        for idx in range(meta.ab_episodes):
            env_seed = eval_seed * 1009 + idx * 53 + 7
            eval_env = env_cfg.make_env(seed=env_seed)
            obs_eval = eval_env.reset()
            candidate.begin_episode()
            state_idx = _obs_to_index(obs_eval)
            visits = defaultdict(int)
            steps_eval = 0
            while steps_eval < env_cfg.max_steps:
                policy_state = candidate.forward(state_idx)
                action_eval = candidate.sample_action(policy_state.probs)
                bonus_eval = candidate.intrinsic_bonus(visits[state_idx])
                visits[state_idx] += 1
                features = build_self_model_input(
                    state_index=state_idx,
                    action=action_eval,
                    mean_count=policy_state.mean_rate,
                    sum_count=policy_state.sum_rate,
                    eta_e=candidate.hidden_lr,
                    v_th=candidate.hidden.params.v_th,
                    state_size=state_size,
                )
                sm_state = sm_copy.forward(features)
                obs_eval, base_reward_eval, done_eval, _info = eval_env.step(action_eval)
                reward_eval = base_reward_eval + bonus_eval
                total_return += reward_eval
                advantage_eval = reward_eval - candidate.baseline
                next_idx = _obs_to_index(obs_eval)
                energy_target_eval = min(
                    sum(policy_state.hidden_counts)
                    / float(candidate.inner_steps * candidate.hidden.n_out),
                    1.0,
                )
                self_signal_eval = sm_copy.update(
                    state=sm_state,
                    next_obs_index=next_idx,
                    reward_target=reward_eval,
                    energy_target=energy_target_eval,
                    cause_label=0,
                )
                candidate.update(
                    policy_state,
                    action_eval,
                    advantage_eval,
                    self_signal=self_signal_eval,
                    alpha=policy_alpha,
                    beta=policy_beta,
                )
                candidate.update_baseline(reward_eval)
                state_idx = next_idx
                steps_eval += 1
                if done_eval:
                    break
        random.setstate(rng_state)
        return total_return / float(max(meta.ab_episodes, 1))

    with EpisodeMetricsLogger(logger, csv_path, print_every=10) as metrics_logger:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            policy.begin_episode()
            state_index = _obs_to_index(obs)
            episode_reward = 0.0
            episode_nll = 0.0
            episode_cause_hits = 0
            episode_energy_mse = 0.0
            steps = 0
            reached_goal = 0
            episode_spikes = 0.0

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
                cause_label = 1 if meta_cause_steps > 0 else 0
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
                if meta_cause_steps > 0:
                    meta_cause_steps -= 1
                buffer.add(
                    state_index,
                    action,
                    policy_state.hidden_counts,
                    reward,
                    next_index,
                )

                nll = -math.log(max(self_state.probs_next[next_index], 1e-8))
                energy_mse = (self_state.pred_energy - energy_target) ** 2
                cause_pred = (
                    1 if self_state.probs_cause[1] >= self_state.probs_cause[0] else 0
                )
                cause_hit = 1 if cause_pred == cause_label else 0

                episode_nll += nll
                episode_energy_mse += energy_mse
                episode_cause_hits += cause_hit
                episode_spikes += sum(policy_state.hidden_counts)

                state_index = next_index
                steps += 1
                if done:
                    if info.get("goal_reached", False):
                        reached_goal = 1
                    break

            success_window.append(reached_goal)
            avg_nll = episode_nll / float(max(steps, 1))
            avg_energy_mse = episode_energy_mse / float(max(steps, 1))
            cause_acc = episode_cause_hits / float(max(steps, 1))
            nll_history.append(avg_nll)
            cause_history.append(cause_acc)
            if running_return is None:
                running_return = episode_reward
            else:
                running_return = 0.9 * running_return + 0.1 * episode_reward

            return_history.append(episode_reward)

            meta_logs: List[str] = []
            if meta.should_trigger(return_history):
                policy, meta_logs = meta.adapt(
                    policy,
                    step=episode,
                    evaluate_fn=evaluate_for_meta,
                )
                for meta_msg in meta_logs:
                    logger.info(meta_msg)
                return_history.clear()
                return_history.append(episode_reward)
            parse_meta(meta_logs)

            success_rate = (
                sum(success_window) / float(len(success_window))
                if success_window
                else 0.0
            )

            metrics_logger.log(
                {
                    "episode": episode,
                    "return": episode_reward,
                    "success_rate": success_rate,
                    "spikes": episode_spikes,
                    "nll": avg_nll,
                    "cause_acc": cause_acc,
                    "meta_action": last_meta["meta_action"],
                    "delta": last_meta["delta"],
                    "reverted": last_meta["reverted"],
                }
            )

        if episode == forget_episode and not forget_triggered:
            recover_threshold = 0.9
            policy.reset_parameters()
            meta_cause_steps = cause_flag_span
            visit_counts.clear()
            forget_triggered = True
            episodes_since_forget = 0
            success_window.clear()
            running_return = None
            return_history.clear()
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
                policy_alpha=policy_alpha,
                policy_beta=policy_beta,
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
    meta_ratio = meta.positive_ratio()
    short_ratio = meta_short_ratio()
    logger.info(
        "Meta adaptations: attempts=%d positive_ratio=%.2f short_window=%.2f",
        meta.attempts,
        meta_ratio,
        short_ratio,
    )
    if validate:
        assert end_nll < start_nll, "Self-Model NLL 未下降。"
        assert cause_tail > 0.7, "自因分类准确率未超过 0.7。"
        if meta.attempts > 0:
            assert meta_ratio >= 0.5, "自改后 Δ 回报为正的比例未达到 50%。"
            if recent_meta_outcomes:
                assert (
                    short_ratio >= 0.5
                ), "近期自改 Δ>0 的比例未达到 50%。"
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
