"""GridWorld 脉冲网络在线训练脚本（e-prop + MetaLearner）。"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.gridworld import GridWorld
from meta.autoadapt import MetaLearner
from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from snn.policy import PolicyHead
from tools.logger import EpisodeMetricsLogger, get_logger, setup_logging


def patched_surrogate(u: float, slope: float = 1.5) -> float:
    """较平滑的自定义替代导数。"""
    denom = 1.0 + slope * abs(u)
    return slope / (denom * denom)


@dataclass
class GridWorldConfig:
    """环境参数容器，便于复用。"""

    size: int = 5
    slip_prob: float = 0.1
    step_cost: float = -0.01
    goal_reward: float = 1.0
    max_steps: int = 50

    def make_env(self, *, seed: int | None = None) -> GridWorld:
        return GridWorld(
            size=self.size,
            slip_prob=self.slip_prob,
            step_cost=self.step_cost,
            goal_reward=self.goal_reward,
            max_steps=self.max_steps,
            seed=seed,
        )


class EpropGridAgent:
    """使用 DenseLIF + PolicyHead 的在线策略。"""

    def __init__(
        self,
        state_size: int,
        *,
        hidden_size: int = 32,
        inner_steps: int = 12,
        eta_e: float = 0.035,
        lam_e: float = 0.9,
        policy_lr: float = 0.06,
        intrinsic_beta: float = 0.35,
        baseline_beta: float = 0.05,
        seed: int | None = None,
    ) -> None:
        eta_e = max(1e-6, eta_e)
        lam_e = max(0.0, min(lam_e, 0.999))
        params = LIFParams(v_th=0.5, tau_m=8.0, tau_a=16.0, beta=0.35, refractory=2)
        self.hidden = DenseLIF(
            n_in=state_size * 2,
            n_out=hidden_size,
            params=params,
            surrogate_fn=fast_sigmoid_surrogate,
            eligibility_lambda=lam_e,
        )
        self.policy = PolicyHead(hidden_size, 4, policy_lr, seed=seed)
        self.eta_e = eta_e
        self.lam_e = lam_e
        self.intrinsic_beta = intrinsic_beta
        self.inner_steps = max(1, inner_steps)
        self.min_inner_steps = 4
        self.max_inner_steps = 30
        self.min_hidden = 8
        self.max_hidden = max(hidden_size * 2, hidden_size + 4)
        self.baseline = 0.0
        self.baseline_beta = baseline_beta
        self.surrogate_name = "fast_sigmoid"
        self._seed = seed or 0
        self._policy_seed = self._seed
        self.reseed(self._seed)
        self.temporal = LinearTemporalUnit(
            n_in=state_size,
            n_state=state_size,
            beta=0.9,
        )
        self._rate_rng = random.Random(seed or 0)
        self.state_size = state_size

    def reseed(self, seed: int) -> None:
        """重置采样随机源，便于 A/B 测试复现。"""
        self._policy_seed = seed
        self.policy._rng.seed(seed)  # noqa: SLF001

    def begin_episode(self) -> None:
        self.hidden.reset_state()
        self.temporal.reset()

    def _encode_obs(self, obs: Sequence[float]) -> List[float]:
        return [1.0 if val >= 0.5 else 0.0 for val in obs]

    def _compose_rates(self, obs: Sequence[float]) -> List[float]:
        obs_bits = self._encode_obs(obs)
        temporal_state = self.temporal.transform(obs_bits)
        temporal_rates = [max(0.0, min(val, 1.0)) for val in temporal_state]
        return obs_bits + temporal_rates

    def _sample_spikes(self, rates: Sequence[float]) -> List[int]:
        spikes: List[int] = []
        for rate in rates:
            if rate >= 1.0:
                spikes.append(1)
            elif rate <= 0.0:
                spikes.append(0)
            else:
                spikes.append(1 if self._rate_rng.random() < rate else 0)
        return spikes

    def _integrate_counts(self, obs: Sequence[float]) -> List[float]:
        counts = [0.0 for _ in range(self.hidden.n_out)]
        rates = self._compose_rates(obs)
        for _ in range(self.inner_steps):
            spikes_in = self._sample_spikes(rates)
            spikes, _psis, _elig, _bias = self.hidden.step(spikes_in)
            for j, fired in enumerate(spikes):
                counts[j] += fired
        scale = 1.0 / float(max(self.inner_steps, 1))
        return [c * scale for c in counts]

    def intrinsic_bonus(self, visits: int) -> float:
        return self.intrinsic_beta / math.sqrt(visits + 1.0)

    def _learning_signal(
        self,
        grad: Sequence[float],
        advantage: float,
        weights_snapshot: Sequence[Sequence[float]],
    ) -> List[float]:
        third = []
        for j in range(self.hidden.n_out):
            accum = 0.0
            for action in range(len(grad)):
                accum += weights_snapshot[action][j] * grad[action]
            third.append(accum * advantage)
        return third

    def learn(
        self,
        counts: Sequence[float],
        probs: Sequence[float],
        action: int,
        reward: float,
    ) -> None:
        advantage = reward - self.baseline
        grad = self.policy.policy_grad(probs, action)
        weights_snapshot = [row[:] for row in self.policy.weights]
        third_factor = self._learning_signal(grad, advantage, weights_snapshot)
        self.policy.update(counts, grad, advantage)
        self.hidden.eprop_apply(third_factor, self.eta_e)
        self.baseline += self.baseline_beta * advantage

    def _append_hidden_neuron(self) -> None:
        for i in range(self.hidden.n_in):
            self.hidden.weights[i].append(random.uniform(-0.1, 0.1))
            self.hidden.eligibility[i].append(0.0)
        self.hidden.bias.append(0.0)
        self.hidden.bias_eligibility.append(0.0)
        self.hidden.n_out += 1
        self.hidden.reset_state()
        self.policy.add_input()

    def _remove_hidden_neuron(self) -> bool:
        if self.hidden.n_out <= self.min_hidden:
            return False
        idx = self.hidden.n_out - 1
        for i in range(self.hidden.n_in):
            self.hidden.weights[i].pop(idx)
            self.hidden.eligibility[i].pop(idx)
        self.hidden.bias.pop(idx)
        self.hidden.bias_eligibility.pop(idx)
        self.hidden.n_out -= 1
        self.hidden.reset_state()
        return self.policy.prune_input()

    def apply_modification(self, action: str) -> Tuple[bool, str | None]:
        """供 MetaLearner 调用的自改接口。"""
        info: str | None = None
        if action == "eta_up":
            self.eta_e = min(self.eta_e * 1.25, 0.12)
            self.policy.lr = min(self.policy.lr * 1.15, 0.12)
            return True, f"eta_e={self.eta_e:.3f}"
        if action == "eta_down":
            self.eta_e = max(self.eta_e * 0.8, 0.005)
            self.policy.lr = max(self.policy.lr * 0.8, 0.02)
            return True, f"eta_e={self.eta_e:.3f}"
        if action == "vth_up":
            self.hidden.params.v_th = min(self.hidden.params.v_th + 0.05, 1.2)
            return True, info
        if action == "vth_down":
            self.hidden.params.v_th = max(self.hidden.params.v_th - 0.05, 0.2)
            return True, info
        if action == "intrinsic_up":
            self.intrinsic_beta = min(self.intrinsic_beta * 1.25, 1.2)
            return True, f"beta={self.intrinsic_beta:.3f}"
        if action == "intrinsic_down":
            self.intrinsic_beta = max(self.intrinsic_beta * 0.75, 0.05)
            return True, f"beta={self.intrinsic_beta:.3f}"
        if action == "inner_up":
            if self.inner_steps >= self.max_inner_steps:
                return False, None
            self.inner_steps = min(self.inner_steps + 2, self.max_inner_steps)
            return True, f"inner_steps={self.inner_steps}"
        if action == "inner_down":
            if self.inner_steps <= self.min_inner_steps:
                return False, None
            self.inner_steps = max(self.inner_steps - 2, self.min_inner_steps)
            return True, f"inner_steps={self.inner_steps}"
        if action == "add_neuron":
            if self.hidden.n_out >= self.max_hidden:
                return False, None
            self._append_hidden_neuron()
            return True, f"n_hidden={self.hidden.n_out}"
        if action == "prune_neuron":
            if not self._remove_hidden_neuron():
                return False, None
            return True, f"n_hidden={self.hidden.n_out}"
        if action == "switch_surrogate":
            if self.surrogate_name == "fast_sigmoid":
                self.hidden.set_surrogate(triangular_surrogate)
                self.surrogate_name = "triangular"
            else:
                self.hidden.set_surrogate(fast_sigmoid_surrogate)
                self.surrogate_name = "fast_sigmoid"
            return True, self.surrogate_name
        if action == "patch_surrogate":
            self.hidden.set_surrogate(patched_surrogate)
            self.surrogate_name = "patched"
            return True, "patched"
        return False, None


def _state_index(obs: Sequence[float]) -> int:
    return max(range(len(obs)), key=lambda idx: obs[idx])


def _run_episode(
    agent: EpropGridAgent,
    env: GridWorld,
    visit_counts: DefaultDict[int, int],
    *,
    training: bool,
) -> Tuple[float, bool, float, float]:
    obs = env.reset()
    agent.begin_episode()
    total_reward = 0.0
    base_reward_sum = 0.0
    goal_reached = False
    steps = 0
    done = False
    spike_sum = 0.0
    while not done and steps < env.max_steps:
        counts = agent._integrate_counts(obs)
        probs = agent.policy.softmax(agent.policy.logits(counts))
        action = agent.policy.sample_action(probs)
        state_idx = _state_index(obs)
        bonus = agent.intrinsic_bonus(visit_counts[state_idx])
        visit_counts[state_idx] += 1
        next_obs, base_reward, done, info = env.step(action)
        reward = base_reward + bonus
        total_reward += reward
        base_reward_sum += base_reward
        spike_sum += sum(counts) * agent.inner_steps
        if training:
            agent.learn(counts, probs, action, reward)
        if info.get("goal_reached", False):
            goal_reached = True
        obs = next_obs
        steps += 1
    return total_reward, goal_reached, spike_sum, base_reward_sum


def _evaluate_agent(
    agent: EpropGridAgent,
    env_cfg: GridWorldConfig,
    *,
    episodes: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    score = 0.0
    for idx in range(episodes):
        env_seed = seed * 997 + idx * 131
        env = env_cfg.make_env(seed=env_seed)
        agent.reseed(rng.randrange(1_000_000))
        visits: DefaultDict[int, int] = collections.defaultdict(int)
        reward, _success, _spikes, _env_return = _run_episode(
            agent,
            env,
            visits,
            training=False,
        )
        score += reward
    return score / float(max(episodes, 1))


def train_gridworld(
    *,
    episodes: int,
    seed: int | None,
    env_cfg: GridWorldConfig,
    eta_e: float = 0.035,
    lam_e: float = 0.9,
    inner_steps: int = 12,
    intrinsic_beta: float = 0.35,
) -> Dict[str, float]:
    if seed is not None:
        random.seed(seed)
    logger = get_logger(__name__)
    agent = EpropGridAgent(
        state_size=env_cfg.size * env_cfg.size,
        inner_steps=inner_steps,
        eta_e=eta_e,
        lam_e=lam_e,
        intrinsic_beta=intrinsic_beta,
        seed=seed,
    )
    meta = MetaLearner(window=20, min_delta=0.05, ab_episodes=5)
    visit_counts: DefaultDict[int, int] = collections.defaultdict(int)
    return_history: List[float] = []
    all_returns: List[float] = []
    rolling_returns: collections.deque[float] = collections.deque(maxlen=10)
    rolling_env_returns: collections.deque[float] = collections.deque(maxlen=10)
    rolling_success: collections.deque[int] = collections.deque(maxlen=10)
    success_history: List[int] = []
    csv_path = pathlib.Path("runs/gridworld_metrics.csv")
    last_meta = {"meta_action": "none", "delta": 0.0, "reverted": False}

    def evaluate_fn(candidate: EpropGridAgent, eval_seed: int) -> float:
        return _evaluate_agent(
            candidate,
            env_cfg,
            episodes=meta.ab_episodes,
            seed=eval_seed,
        )

    def parse_meta(logs: List[str]) -> None:
        nonlocal last_meta
        if not logs:
            last_meta = {"meta_action": "none", "delta": 0.0, "reverted": False}
            return
        message = logs[-1]
        try:
            action = message.split("action=")[1].split()[0]
        except (IndexError, ValueError):
            action = "unknown"
        try:
            delta_str = message.split("delta=")[1].split()[0]
            delta_val = float(delta_str)
        except (IndexError, ValueError):
            delta_val = 0.0
        reverted = "reverted" in message
        last_meta = {
            "meta_action": action,
            "delta": delta_val,
            "reverted": reverted,
        }

    with EpisodeMetricsLogger(logger, csv_path, print_every=10) as metrics_logger:
        for episode in range(1, episodes + 1):
            env_seed = (seed or 0) * 1009 + episode * 47 + 17
            env = env_cfg.make_env(seed=env_seed)
            agent.reseed(env_seed)
            reward, success, spikes, env_return = _run_episode(
                agent,
                env,
                visit_counts,
                training=True,
            )
            return_history.append(reward)
            all_returns.append(reward)
            rolling_returns.append(reward)
            rolling_success.append(1 if success else 0)
            success_history.append(1 if success else 0)
            rolling_env_returns.append(env_return)

            meta_logs: List[str] = []
            if meta.should_trigger(return_history):
                agent, meta_logs = meta.adapt(
                    agent,
                    step=episode,
                    evaluate_fn=evaluate_fn,
                )
                for log_line in meta_logs:
                    logger.info(log_line)
                return_history.clear()
            parse_meta(meta_logs)

            success_rate = (
                sum(rolling_success) / float(len(rolling_success))
                if rolling_success
                else 0.0
            )

            metrics_logger.log(
                {
                    "episode": episode,
                    "return": reward,
                    "success_rate": success_rate,
                    "spikes": spikes,
                    "nll": 0.0,
                    "cause_acc": 0.0,
                    "meta_action": last_meta["meta_action"],
                    "delta": last_meta["delta"],
                    "reverted": last_meta["reverted"],
                }
            )

    window = min(20, len(success_history))
    final_success = sum(success_history[-window:]) / float(max(window, 1))
    tail_returns = all_returns[-window:] if window else []
    mean_return = sum(tail_returns) / float(max(len(tail_returns), 1)) if tail_returns else 0.0
    logger.info(
        "Training finished: success@tail %.2f positive_ratio %.2f",
        final_success,
        meta.positive_ratio(),
    )
    return {
        "tail_success": final_success,
        "tail_return": mean_return,
        "positive_ratio": meta.positive_ratio(),
    }


def train_once(
    episodes: int = 10,
    seed: int | None = None,
    *,
    env_cfg: GridWorldConfig | None = None,
) -> Tuple[float, float, float]:
    """Lightweight train loop used by background daemons."""
    env_cfg = env_cfg or GridWorldConfig()
    agent = EpropGridAgent(
        state_size=env_cfg.size * env_cfg.size,
        inner_steps=12,
        eta_e=0.035,
        lam_e=0.9,
        intrinsic_beta=0.35,
        seed=seed,
    )
    visit_counts: DefaultDict[int, int] = collections.defaultdict(int)
    total_reward = 0.0
    total_spikes = 0.0
    successes = 0
    for episode in range(1, episodes + 1):
        env_seed = (seed or 0) * 2029 + episode * 131
        env = env_cfg.make_env(seed=env_seed)
        agent.reseed(env_seed)
        reward, success, spikes, _ = _run_episode(
            agent,
            env,
            visit_counts,
            training=True,
        )
        total_reward += reward
        total_spikes += spikes
        if success:
            successes += 1
    count = float(max(episodes, 1))
    avg_return = total_reward / count
    success_rate = successes / count
    avg_spikes = total_spikes / count
    return avg_return, success_rate, avg_spikes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridWorld 在线 e-prop 训练。")
    parser.add_argument("--episodes", type=int, default=80, help="训练回合数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--slip-prob", type=float, default=0.1, help="动作随机滑移概率")
    parser.add_argument("--step-cost", type=float, default=-0.01, help="每步惩罚")
    parser.add_argument("--goal-reward", type=float, default=1.0, help="终点奖励")
    parser.add_argument("--max-steps", type=int, default=50, help="单回合最大步数")
    parser.add_argument("--eta-e", type=float, default=0.035, help="e-prop 学习率 η_e")
    parser.add_argument("--lam-e", type=float, default=0.9, help="资格迹衰减 λ_e")
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=12,
        help="同一观测积分步骤数",
    )
    parser.add_argument(
        "--intrinsic-beta",
        type=float,
        default=0.35,
        help="探索新奇奖励系数 β",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = get_logger(__name__)
    env_cfg = GridWorldConfig(
        slip_prob=args.slip_prob,
        step_cost=args.step_cost,
        goal_reward=args.goal_reward,
        max_steps=args.max_steps,
    )
    metrics = train_gridworld(
        episodes=args.episodes,
        seed=args.seed,
        env_cfg=env_cfg,
        eta_e=args.eta_e,
        lam_e=args.lam_e,
        inner_steps=args.inner_steps,
        intrinsic_beta=args.intrinsic_beta,
    )
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
