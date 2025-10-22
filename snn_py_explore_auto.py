"""GridWorld 主动探索 + REINFORCE + 自适应 MetaLearner 原型。"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from meta.autoadapt import MetaLearner
from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from snn.selfmodel import SelfModel
from tools.logger import get_logger, setup_logging
import ast


Action = int  # 0:上, 1:下, 2:左, 3:右


def patched_surrogate(u: float, gain: float = 1.5) -> float:
    """基于代码热补丁的替代导数，提供更窄的梯度窗口。"""
    denom = 1.0 + gain * u * u
    return gain / (denom * denom)


class CodePatcher:
    """运行时构建替代导数函数，并应用到所有神经元。"""

    DEFAULT_EXPRESSIONS = [
        "(1.0 - math.fabs(v) / (w if w > 1e-6 else 1e-6)) if math.fabs(v) <= w else 0.0",
        "1.0 if math.fabs(v) <= w else 0.0",
        "g / ((1.0 + g * math.fabs(v)) ** 2)",
    ]

    def __init__(self, expressions: List[str] | None = None) -> None:
        self.expressions = expressions[:] if expressions else self.DEFAULT_EXPRESSIONS[:]
        self.current_idx = -1
        self.last_patch_info: str | None = None
        self._validate_all()

    def _validate_all(self) -> None:
        for expr in self.expressions:
            self._validate_expression(expr)

    def _validate_expression(self, expr: str) -> None:
        tree = ast.parse(expr, mode="eval")
        allowed = {"v", "w", "g", "math"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise ValueError(f"表达式包含非法标识符: {node.id}")

    def _build_function(self, expr: str):
        namespace = {"math": math}
        code = (
            "def dynamic_surrogate(v, w=1.0, g=2.0):\n"
            f"    return {expr}\n"
        )
        exec(code, namespace, namespace)
        return namespace["dynamic_surrogate"]

    def apply(self, layer: DenseLIF, idx: int | None = None) -> str:
        if not self.expressions:
            raise ValueError("无可用代码热补丁表达式。")
        if idx is None:
            idx = (self.current_idx + 1) % len(self.expressions)
        expr = self.expressions[idx]
        func = self._build_function(expr)
        layer.set_surrogate(func)
        self.current_idx = idx
        self.last_patch_info = f"patch surrogate idx={idx}"
        return self.last_patch_info


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


def build_self_model_input(
    state_index: int,
    action: int,
    mean_rate: float,
    sum_rate: float,
    eta_e: float,
    v_th: float,
    state_size: int,
) -> List[float]:
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
        self.code_patcher = CodePatcher()
        self.last_patch_info: str | None = None

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
            clip_value(err * advantage, self.clip) for err in policy_error
        ]
        learning_signals = []
        for h in range(self.hidden.n_out):
            signal = 0.0
            for a in range(4):
                signal += policy_error[a] * self.readout_weights[h][a]
            learning_signals.append(clip_value(signal, self.clip))
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
                combined.append(clip_value(combined_signal, self.clip))
            learning_signals = combined
        else:
            learning_signals = [clip_value(alpha * sig, self.clip) for sig in learning_signals]
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
            return self.add_neuron(), info
        if action == "prune_neuron":
            return self.prune_neuron(), info
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
            self.surrogate_name = f"dynamic_{self.code_patcher.current_idx}"
            self.last_patch_info = info
            return True, info
        return False, info


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
    self_params = LIFParams(v_th=0.5, tau_m=10.0, tau_a=20.0, beta=0.35, refractory=2)
    self_model = SelfModel(
        obs_dim=env_cfg.size * env_cfg.size,
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
        state = env.reset()
        state_index = env.state_index(state)
        episode_reward = 0.0
        reached_goal = 0
        episode_nll = 0.0
        episode_cause_hits = 0
        episode_energy_mse = 0.0
        steps = 0
        while steps < env.cfg.max_steps:
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
                state_size=env_cfg.size * env_cfg.size,
            )
            self_state = self_model.forward(features)

            next_state, base_reward, done = env.step(action)
            reward = base_reward + bonus
            episode_reward += reward
            advantage = reward - agent.baseline
            next_index = env.state_index(next_state)

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

            state = next_state
            state_index = next_index
            steps += 1
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
            def evaluate_for_meta(candidate_agent: SNNAgent, seed: int) -> float:
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
    assert overall_success > 0.6, "终点成功率未超过 60%。"
    if cause_history:
        assert (
            sum(cause_history[-20:]) / float(min(20, len(cause_history)))
            >= 0.7
        ), "自因分类准确率未达到 0.7。"
    assert meta.positive_ratio() >= 0.5, "自改 Δ>0 的比例未达到 50%。"


if __name__ == "__main__":
    run_training()
