"""GridWorld + DenseLIF + PolicyHead 训练脚本。"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
from typing import Dict, List, Sequence, Tuple

# 将项目根目录加入路径，便于脚本独立运行。
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.gridworld import GridWorld
from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate
from snn.policy import PolicyHead


def encode_poisson(
    obs: Sequence[float],
    rng: random.Random,
    high_rate: float,
    low_rate: float,
) -> List[int]:
    """将 one-hot 观察编码为泊松脉冲。"""
    spikes: List[int] = []
    for value in obs:
        rate = high_rate if value > 0.5 else low_rate
        spikes.append(1 if rng.random() < rate else 0)
    return spikes


def accumulate_hidden_activity(
    layer: DenseLIF,
    obs: Sequence[float],
    inner_steps: int,
    rng: random.Random,
    high_rate: float,
    low_rate: float,
) -> Tuple[List[float], List[int]]:
    """对同一观测积分 multiple inner steps，返回平均脉冲率和累计发放。"""
    layer.reset_state()
    counts = [0 for _ in range(layer.n_out)]
    for _ in range(inner_steps):
        spikes, _, _, _ = layer.step(encode_poisson(obs, rng, high_rate, low_rate))
        counts = [c + s for c, s in zip(counts, spikes)]
    rates = [c / float(inner_steps) for c in counts]
    return rates, counts


def sample_from_probs(probs: Sequence[float], rng: random.Random) -> int:
    """按概率分布采样动作。"""
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def compute_learning_signal(
    readout_weights: Sequence[Sequence[float]],
    scaled_grad: Sequence[float],
) -> List[float]:
    """将读出层梯度传播为隐藏层学习信号。"""
    n_hidden = len(readout_weights[0])
    signals = [0.0 for _ in range(n_hidden)]
    for a, grad in enumerate(scaled_grad):
        for h in range(n_hidden):
            signals[h] += grad * readout_weights[a][h]
    return signals


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    rng = random.Random(args.seed + 17)

    env = GridWorld()
    obs = env.reset()
    state_dim = len(obs)

    params = LIFParams(
        v_th=args.v_th,
        tau_m=args.tau_m,
        tau_a=args.tau_a,
        beta=args.beta_a,
        refractory=args.refractory,
    )
    hidden = DenseLIF(
        n_in=state_dim,
        n_out=args.hidden_size,
        params=params,
        surrogate_fn=fast_sigmoid_surrogate,
    )
    head = PolicyHead(
        n_in=args.hidden_size,
        n_actions=4,
        lr=args.head_lr,
    )

    baseline = 0.0
    visit_counts: Dict[int, int] = {}
    returns: List[float] = []
    successes: List[int] = []
    best_avg = None

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        total_return = 0.0
        done = False
        reached_goal = 0

        while not done:
            state_index = max(range(len(obs)), key=lambda idx: obs[idx])
            visits = visit_counts.get(state_index, 0)
            bonus = args.intrinsic_beta / math.sqrt(visits + 1)
            visit_counts[state_index] = visits + 1

            rates, _ = accumulate_hidden_activity(
                hidden,
                obs,
                args.inner_steps,
                rng,
                args.high_rate,
                args.low_rate,
            )
            logits = head.logits(rates)
            probs = head.softmax(logits)
            action = sample_from_probs(probs, rng)

            next_obs, reward, done, info = env.step(action)
            shaped_reward = reward + bonus
            total_return += shaped_reward

            advantage = shaped_reward - baseline
            baseline = (
                (1.0 - args.baseline_beta) * baseline
                + args.baseline_beta * shaped_reward
            )

            readout_weights = [row[:] for row in head.weights]
            policy_grad = head.policy_grad(rates, action, probs)
            head.update(rates, policy_grad, advantage)

            scaled_grad = [g * advantage for g in policy_grad]
            learning_signal = compute_learning_signal(readout_weights, scaled_grad)
            hidden.eprop_apply(learning_signal, args.eta_e)
            hidden.reset_state()

            obs = next_obs
            if done and info.get("goal_reached", False):
                reached_goal = 1

        returns.append(total_return)
        successes.append(reached_goal)

        if episode % args.log_interval == 0:
            window_returns = returns[-args.log_interval :]
            avg_return = sum(window_returns) / len(window_returns)
            if best_avg is None or avg_return > best_avg:
                best_avg = avg_return
            window_success = successes[-args.log_interval :]
            avg_success = sum(window_success) / len(window_success)
            print(
                f"[train] ep {episode:03d} avg_return={best_avg:.3f} "
                f"window_return={avg_return:.3f} success={avg_success:.2%}"
            )

    final_success = sum(successes[-min(20, len(successes)) :]) / float(
        min(20, len(successes)) or 1
    )
    print(
        f"[train] final_success={final_success:.2%} "
        f"(episodes={args.episodes}, inner_steps={args.inner_steps})"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridWorld SNN 训练脚本")
    parser.add_argument("--episodes", type=int, default=80, help="训练回合数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--hidden-size", type=int, default=32, help="隐藏神经元数量")
    parser.add_argument("--inner-steps", type=int, default=16, help="每步内部积分次数")
    parser.add_argument("--eta-e", type=float, default=0.12, help="e-prop 学习率")
    parser.add_argument("--head-lr", type=float, default=0.25, help="策略头学习率")
    parser.add_argument("--intrinsic-beta", type=float, default=0.25, help="内在动机系数 β")
    parser.add_argument("--baseline-beta", type=float, default=0.05, help="基线平滑系数")
    parser.add_argument("--high-rate", type=float, default=0.9, help="one-hot=1 的发放率")
    parser.add_argument("--low-rate", type=float, default=0.05, help="one-hot=0 的发放率")
    parser.add_argument("--v-th", type=float, default=0.52, help="LIF 阈值")
    parser.add_argument("--tau-m", type=float, default=9.0, help="膜时间常数")
    parser.add_argument("--tau-a", type=float, default=18.0, help="适应时间常数")
    parser.add_argument("--beta-a", type=float, default=0.35, help="适应强度 β")
    parser.add_argument("--refractory", type=int, default=2, help="不应期步数")
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
