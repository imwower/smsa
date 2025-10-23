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
from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from snn.policy import PolicyHead
from tools.logger import get_logger, setup_logging


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
        hidden_lr: float = 0.035,
        policy_lr: float = 0.06,
        intrinsic_beta: float = 0.35,
        baseline_beta: float = 0.05,
        seed: int | None = None,
    ) -> None:
        params = LIFParams(v_th=0.5, tau_m=8.0, tau_a=16.0, beta=0.35, refractory=2)
        self.hidden = DenseLIF(
            n_in=state_size,
            n_out=hidden_size,
            params=params,
            surrogate_fn=fast_sigmoid_surrogate,
        )
        self.policy = PolicyHead(hidden_size, 4, policy_lr, seed=seed)
        self.hidden_lr = hidden_lr
        self.intrinsic_beta = intrinsic_beta
        self.inner_steps = inner_steps
        self.baseline = 0.0
        self.baseline_beta = baseline_beta
        self.surrogate_name = "fast_sigmoid"
        self._seed = seed or 0
        self._policy_seed = self._seed
        self.reseed(self._seed)

    def reseed(self, seed: int) -> None:
        """重置采样随机源，便于 A/B 测试复现。"""
        self._policy_seed = seed
        self.policy._rng.seed(seed)  # noqa: SLF001

    def begin_episode(self) -> None:
        self.hidden.reset_state()

    def _integrate_counts(self, obs: Sequence[float]) -> List[float]:
        counts = [0.0 for _ in range(self.hidden.n_out)]
        spikes_in = [int(round(x)) for x in obs]
        for _ in range(self.inner_steps):
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
        self.hidden.eprop_apply(third_factor, self.hidden_lr)
        self.baseline += self.baseline_beta * advantage

    def apply_modification(self, action: str) -> Tuple[bool, str | None]:
        """供 MetaLearner 调用的自改接口。"""
        info: str | None = None
        if action == "eta_up":
            self.hidden_lr = min(self.hidden_lr * 1.25, 0.12)
            self.policy.lr = min(self.policy.lr * 1.15, 0.12)
            return True, info
        if action == "eta_down":
            self.hidden_lr = max(self.hidden_lr * 0.8, 0.01)
            self.policy.lr = max(self.policy.lr * 0.8, 0.02)
            return True, info
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
            if self.inner_steps >= 24:
                return False, None
            self.inner_steps += 2
            return True, f"inner_steps={self.inner_steps}"
        if action == "inner_down":
            if self.inner_steps <= 6:
                return False, None
            self.inner_steps -= 2
            return True, f"inner_steps={self.inner_steps}"
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
) -> Tuple[float, bool]:
    obs = env.reset()
    agent.begin_episode()
    total_reward = 0.0
    goal_reached = False
    steps = 0
    done = False
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
        if training:
            agent.learn(counts, probs, action, reward)
        if info.get("goal_reached", False):
            goal_reached = True
        obs = next_obs
        steps += 1
    return total_reward, goal_reached


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
        reward, _ = _run_episode(agent, env, visits, training=False)
        score += reward
    return score / float(max(episodes, 1))


def train_gridworld(
    *,
    episodes: int,
    seed: int | None,
    env_cfg: GridWorldConfig,
) -> Dict[str, float]:
    if seed is not None:
        random.seed(seed)
    logger = get_logger(__name__)
    agent = EpropGridAgent(state_size=env_cfg.size * env_cfg.size, seed=seed)
    meta = MetaLearner(window=20, min_delta=0.05, ab_episodes=5)
    visit_counts: DefaultDict[int, int] = collections.defaultdict(int)
    return_history: List[float] = []
    all_returns: List[float] = []
    rolling_returns: collections.deque[float] = collections.deque(maxlen=10)
    rolling_success: collections.deque[int] = collections.deque(maxlen=10)
    success_history: List[int] = []

    def evaluate_fn(candidate: EpropGridAgent, eval_seed: int) -> float:
        return _evaluate_agent(
            candidate,
            env_cfg,
            episodes=meta.ab_episodes,
            seed=eval_seed,
        )

    for episode in range(1, episodes + 1):
        env_seed = (seed or 0) * 1009 + episode * 47 + 17
        env = env_cfg.make_env(seed=env_seed)
        agent.reseed(env_seed)
        reward, success = _run_episode(agent, env, visit_counts, training=True)
        return_history.append(reward)
        all_returns.append(reward)
        rolling_returns.append(reward)
        rolling_success.append(1 if success else 0)
        success_history.append(1 if success else 0)

        if episode % 10 == 0:
            avg_return = sum(rolling_returns) / float(len(rolling_returns))
            avg_success = sum(rolling_success) / float(len(rolling_success))
            logger.info(
                "Episode %03d avg_return %.3f success_rate %.2f",
                episode,
                avg_return,
                avg_success,
            )

        if meta.should_trigger(return_history):
            agent, meta_logs = meta.adapt(agent, step=episode, evaluate_fn=evaluate_fn)
            for log_line in meta_logs:
                logger.info(log_line)
            return_history.clear()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridWorld 在线 e-prop 训练。")
    parser.add_argument("--episodes", type=int, default=80, help="训练回合数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--slip-prob", type=float, default=0.1, help="动作随机滑移概率")
    parser.add_argument("--step-cost", type=float, default=-0.01, help="每步惩罚")
    parser.add_argument("--goal-reward", type=float, default=1.0, help="终点奖励")
    parser.add_argument("--max-steps", type=int, default=50, help="单回合最大步数")
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
    metrics = train_gridworld(episodes=args.episodes, seed=args.seed, env_cfg=env_cfg)
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
