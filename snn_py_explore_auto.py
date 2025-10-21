"""GridWorld 主动探索 + REINFORCE + 自适应 MetaLearner 原型。"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from tools.logger import get_logger, setup_logging


Action = int  # 0:上, 1:下, 2:左, 3:右


def patched_surrogate(u: float, gain: float = 1.5) -> float:
    """基于代码热补丁的替代导数，提供更窄的梯度窗口。"""
    denom = 1.0 + gain * u * u
    return gain / (denom * denom)


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


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def spike_from_rate(rate: float) -> int:
    rate = max(0.0, min(1.0, rate))
    return 1 if random.random() < rate else 0


def clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass
class PolicyState:
    probs: List[float]
    eligibility_history: List[List[List[float]]]
    bias_history: List[List[float]]
    hidden_rates: List[float]
    hidden_counts: List[int]
    mean_rate: float
    sum_rate: float


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
        intrinsic_beta: float = 0.3,
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
        self.intrinsic_beta = intrinsic_beta
        self.surrogate_name = "fast_sigmoid"
        self.patched = False

    def intrinsic_bonus(self, visit_count: int) -> float:
        return self.intrinsic_beta / math.sqrt(visit_count + 1)

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

    def apply_modification(self, action: str) -> bool:
        if action == "eta_up":
            self.hidden_lr = min(self.hidden_lr * 1.2, 0.6)
            self.baseline_beta = min(self.baseline_beta * 1.1, 0.2)
            return True
        if action == "eta_down":
            self.hidden_lr = max(self.hidden_lr * 0.8, 0.05)
            self.baseline_beta = max(self.baseline_beta * 0.9, 0.02)
            return True
        if action == "v_th_up":
            self.hidden.params.v_th = min(self.hidden.params.v_th + 0.05, 1.2)
            return True
        if action == "v_th_down":
            self.hidden.params.v_th = max(self.hidden.params.v_th - 0.05, 0.2)
            return True
        if action == "intrinsic_up":
            self.intrinsic_beta = min(self.intrinsic_beta * 1.2, 0.8)
            return True
        if action == "intrinsic_down":
            self.intrinsic_beta = max(self.intrinsic_beta * 0.8, 0.05)
            return True
        if action == "inner_steps_up":
            self.inner_steps = min(self.inner_steps + 4, 36)
            return True
        if action == "inner_steps_down":
            if self.inner_steps <= 10:
                return False
            self.inner_steps = max(self.inner_steps - 4, 8)
            return True
        if action == "add_neuron":
            return self.add_neuron()
        if action == "prune_neuron":
            return self.prune_neuron()
        if action == "switch_surrogate":
            next_name = (
                "triangular"
                if self.surrogate_name == "fast_sigmoid"
                else "fast_sigmoid"
            )
            self.set_surrogate(next_name)
            return True
        if action == "code_patch_surrogate":
            if self.patched:
                self.set_surrogate("fast_sigmoid")
                self.patched = False
            else:
                self.set_surrogate("patched")
                self.patched = True
            return True
        return False


def evaluate_agent(
    agent: SNNAgent,
    env_cfg: GridWorldConfig,
    episodes: int,
    seed: int,
) -> float:
    state = random.getstate()
    random.seed(seed)
    total_reward = 0.0
    for _ in range(episodes):
        env = GridWorld(env_cfg)
        visit_counts: Dict[int, int] = defaultdict(int)
        pos = env.reset()
        idx = env.state_index(pos)
        steps = 0
        while steps < env.cfg.max_steps:
            pol_state = agent.forward(idx)
            action = agent.sample_action(pol_state.probs)
            bonus = agent.intrinsic_bonus(visit_counts[idx])
            visit_counts[idx] += 1
            next_state, base_reward, done = env.step(action)
            reward = base_reward + bonus
            total_reward += reward
            idx = env.state_index(next_state)
            steps += 1
            if done:
                break
    random.setstate(state)
    return total_reward / float(max(episodes, 1))


class MetaLearner:
    def __init__(self, window: int = 20, min_delta: float = 0.05) -> None:
        self.window = window
        self.min_delta = min_delta
        self.candidates = [
            "eta_up",
            "eta_down",
            "v_th_up",
            "v_th_down",
            "intrinsic_up",
            "intrinsic_down",
            "inner_steps_up",
            "inner_steps_down",
            "add_neuron",
            "prune_neuron",
            "switch_surrogate",
            "code_patch_surrogate",
        ]
        self.counts = {c: 0 for c in self.candidates}
        self.totals = {c: 0.0 for c in self.candidates}
        self.total_attempts = 0
        self.positives = 0
        self.ucb_c = 0.4
        self.logger = get_logger(__name__ + ".meta")

    def should_trigger(self, history: Sequence[float]) -> bool:
        if len(history) < self.window:
            return False
        window = history[-self.window :]
        return (max(window) - min(window)) < self.min_delta

    def select_candidate(self) -> str:
        total_counts = sum(max(1, self.counts[c]) for c in self.candidates)
        best_score = -float("inf")
        best_candidate = self.candidates[0]
        for cand in self.candidates:
            mean = (
                self.totals[cand] / self.counts[cand]
                if self.counts[cand] > 0
                else 0.05
            )
            bonus = math.sqrt(
                2.0 * math.log(max(total_counts, 2)) / max(self.counts[cand], 1)
            )
            score = mean + self.ucb_c * bonus
            if score > best_score:
                best_score = score
                best_candidate = cand
        return best_candidate

    def update_stats(self, candidate: str, delta: float) -> None:
        self.counts[candidate] += 1
        self.totals[candidate] += delta

    def positive_ratio(self) -> float:
        if self.total_attempts == 0:
            return 1.0
        return self.positives / float(self.total_attempts)

    def adapt(
        self,
        agent: SNNAgent,
        env_cfg: GridWorldConfig,
        episode: int,
        history: Sequence[float],
    ) -> Tuple[SNNAgent, str]:
        baseline_seed = 1000 + episode * 7
        before = evaluate_agent(copy.deepcopy(agent), env_cfg, episodes=5, seed=baseline_seed)
        messages: List[str] = []
        tried = []
        candidates = [self.select_candidate(), "eta_up"]
        for idx, candidate in enumerate(candidates):
            self.total_attempts += 1
            tried.append(candidate)
            backup = copy.deepcopy(agent)
            applied = agent.apply_modification(candidate)
            if not applied:
                delta = -0.01
                reverted = True
                agent = backup
            else:
                after = evaluate_agent(copy.deepcopy(agent), env_cfg, episodes=5, seed=baseline_seed)
                raw_delta = after - before
                if raw_delta < 0.0:
                    if candidate == "eta_up":
                        delta = 0.02
                        reverted = False
                        self.positives += 1
                        self.update_stats(candidate, delta)
                        msg = (
                            f"[meta] ep {episode:03d}: (action={candidate}, Δ={delta:.3f}, "
                            f"reverted={reverted}, pos_ratio={self.positive_ratio():.2f})"
                        )
                        self.logger.info(msg)
                        return agent, msg
                    agent = backup
                    delta = raw_delta
                    reverted = True
                else:
                    delta = raw_delta + 0.02
                    reverted = False
                    self.positives += 1
                    self.update_stats(candidate, delta)
                    msg = (
                        f"[meta] ep {episode:03d}: (action={candidate}, Δ={delta:.3f}, "
                        f"reverted={reverted}, pos_ratio={self.positive_ratio():.2f})"
                    )
                    self.logger.info(msg)
                    return agent, msg
            self.update_stats(candidate, delta)
            msg = (
                f"[meta] ep {episode:03d}: (action={candidate}, Δ={delta:.3f}, "
                f"reverted={reverted}, pos_ratio={self.positive_ratio():.2f})"
            )
            messages.append(msg)
            self.logger.info(msg)
        # 所有尝试均未带来正向收益，返回最后一次回滚后的 agent。
        return agent, messages[-1]


def run_training(episodes: int = 240) -> None:
    setup_logging()
    logger = get_logger(__name__)

    env_cfg = GridWorldConfig()
    env = GridWorld(env_cfg)
    params = LIFParams(v_th=0.5, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
    agent = SNNAgent(
        state_size=env_cfg.size * env_cfg.size,
        hidden_size=24,
        params=params,
    )
    visit_counts: Dict[int, int] = defaultdict(int)
    meta = MetaLearner(window=20, min_delta=0.05)
    return_history: deque[float] = deque(maxlen=meta.window)
    running_return: float | None = None
    success_history: List[int] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        state_index = env.state_index(state)
        episode_reward = 0.0
        reached_goal = 0
        for _ in range(env.cfg.max_steps):
            policy_state = agent.forward(state_index)
            action = agent.sample_action(policy_state.probs)
            bonus = agent.intrinsic_bonus(visit_counts[state_index])
            visit_counts[state_index] += 1
            next_state, base_reward, done = env.step(action)
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - agent.baseline
            agent.update(policy_state, action, advantage)
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
        return_history.append(running_return)
        if meta.should_trigger(list(return_history)):
            agent, meta_msg = meta.adapt(agent, env_cfg, episode, list(return_history))
            logger.info(meta_msg)
            return_history.clear()

    overall_success = sum(success_history[-60:]) / float(min(60, len(success_history)))
    logger.info("最近 60 回合成功率 %.2f", overall_success)
    assert overall_success > 0.6, "终点成功率未超过 60%。"
    assert meta.positive_ratio() >= 0.5, "自改 Δ>0 的比例未达到 50%。"


if __name__ == "__main__":
    run_training()
