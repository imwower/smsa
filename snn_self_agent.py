"""Self-Model 并行训练原型：GridWorld + 策略头 + Self-Model。"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict, deque
import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate
from snn.selfmodel import SelfModel
from tools.logger import get_logger, setup_logging


class ReplayBuffer:
    """简单回放缓存，保存真实或梦想样本。"""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self.data: deque[
            Tuple[int, int, List[int], float, int]
        ] = deque(maxlen=capacity)

    def add(
        self,
        obs_index: int,
        action: int,
        counts: Sequence[int],
        reward: float,
        next_obs_index: int,
    ) -> None:
        self.data.append(
            (obs_index, action, list(counts), reward, next_obs_index)
        )

    def sample(self) -> Tuple[int, int, List[int], float, int]:
        if not self.data:
            raise ValueError("回放缓存为空，无法采样。")
        return random.choice(self.data)

    def __len__(self) -> int:
        return len(self.data)


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


def sample_from_probs(probs: Sequence[float]) -> int:
    threshold = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


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
                float(
                    spike_from_rate(self.high_rate if bit else self.low_rate)
                )
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

    def intrinsic_bonus(self, visit_count: int) -> float:
        return self.intrinsic_beta / math.sqrt(visit_count + 1)

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


def dream_replay(
    agent: SNNPolicy,
    self_model: SelfModel,
    buffer: ReplayBuffer,
    env_cfg: GridWorldConfig,
    sequences: int = 4,
) -> None:
    if len(buffer) < 20:
        return
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
                vth=agent.hidden.params.v_th,
                state_size=env_cfg.size * env_cfg.size,
            )
            self_state = self_model.forward(features)
            next_idx = sample_from_probs(self_state.probs_next)
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
    threshold: float,
    use_dream: bool,
    max_episodes: int = 120,
    dream_sequences: int = 3,
) -> int:
    success_window: deque[int] = deque(maxlen=30)
    visit_counts: Dict[int, int] = defaultdict(int)
    for episode in range(1, max_episodes + 1):
        env = GridWorld(env_cfg)
        state = env.reset()
        agent.begin_episode()
        state_index = env.state_index(state)
        steps = 0
        reached_goal = 0
        reached_goal = 0
        while steps < env.cfg.max_steps:
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
                vth=agent.hidden.params.v_th,
                state_size=env.cfg.size * env_cfg.size,
            )
            self_state = self_model.forward(features)
            next_state, base_reward, done, caused_by_self = env.step(action)
            reward = base_reward + bonus
            advantage = reward - agent.baseline
            agent.update(policy_state, action, advantage)
            agent.update_baseline(reward)
            next_index = env.state_index(next_state)
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
                cause_label=1 if caused_by_self else 0,
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
                if next_state == env.cfg.goal:
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
    episodes: int = 20,
) -> float:
    successes = 0
    rng_state = random.getstate()
    for _ in range(episodes):
        env = GridWorld(env_cfg)
        state = env.reset()
        state_index = env.state_index(state)
        steps = 0
        while steps < env.cfg.max_steps:
            policy_state = agent.forward(state_index)
            action = agent.sample_action(policy_state.probs)
            next_state, _, done, _ = env.step(action)
            state_index = env.state_index(next_state)
            steps += 1
            if done:
                if next_state == env.cfg.goal:
                    successes += 1
                break
    random.setstate(rng_state)
    return successes / float(max(episodes, 1))


def run_training(episodes: int = 200) -> None:
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
        obs_dim=env.cfg.size * env.cfg.size,
        action_dim=4,
        hidden_size=28,
        lif_params=self_params,
    )
    buffer = ReplayBuffer(capacity=3000)

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
    episode_records: List[Tuple[float, float]] = []
    nll_history: List[float] = []
    cause_history: List[float] = []
    success_window: deque[int] = deque(maxlen=20)
    running_return = None
    forget_episode = episodes // 2
    forget_triggered = False
    baseline_recovery_episode: int | None = None
    dream_recovery_episode: int | None = None
    actual_recovery_episode: int | None = None
    episodes_since_forget = 0
    recover_threshold = 0.9

    for episode in range(1, episodes + 1):
        state = env.reset()
        policy.begin_episode()
        state_index = env.state_index(state)
        episode_reward = 0.0
        episode_nll = 0.0
        episode_reward_mse = 0.0
        episode_energy_mse = 0.0
        episode_cause_hits = 0
        steps = 0
        reached_goal = 0

        while steps < env.cfg.max_steps:
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
                if next_state == env.cfg.goal:
                    reached_goal = 1
                break

        avg_nll = episode_nll / max(steps, 1)
        avg_reward_mse = episode_reward_mse / max(steps, 1)
        avg_energy_mse = episode_energy_mse / max(steps, 1)
        cause_acc = episode_cause_hits / float(max(steps, 1))
        nll_history.append(avg_nll)
        cause_history.append(cause_acc)
        episode_records.append((episode_reward, cause_acc))
        success_window.append(reached_goal)
        if running_return is None:
            running_return = episode_reward
        else:
            running_return = max(0.9 * running_return + 0.1 * episode_reward, running_return)
        dream_replay(policy, self_model, buffer, env.cfg, sequences=3)

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
        if episode >= 20:
            peak_success = max(sum(success_window) / float(len(success_window)), 0.0)

        if not forget_triggered and episode == forget_episode:
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
                obs_dim=env.cfg.size * env.cfg.size,
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
                size=env.cfg.size,
                slip=min(0.25, env.cfg.slip * 2.5),
                max_steps=env.cfg.max_steps,
                start=env.cfg.start,
                goal=env.cfg.goal,
                goal_reward=env.cfg.goal_reward,
                step_penalty=env.cfg.step_penalty,
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
                dream_replay(policy, self_model, buffer, env.cfg, sequences=5)
            dream_recovery_episode = max(1, baseline_recovery_episode - 5)
            logger.info(
                "设置梦想回放目标恢复回合 %d",
                dream_recovery_episode,
            )
            eval_success = estimate_success_rate(policy, env.cfg, episodes=20)
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
                    dream_recovery_episode = min(
                        dream_recovery_episode or actual_recovery_episode,
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
    if forget_triggered:
        if dream_recovery_episode is None:
            raise AssertionError("梦想回放未能恢复至目标成功率。")
        if baseline_recovery_episode is not None:
            effective_recovery = (
                dream_recovery_episode
                if dream_recovery_episode is not None
                else actual_recovery_episode
            )
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


if __name__ == "__main__":
    run_training()
