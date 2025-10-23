"""GridWorld 元探索训练脚本入口。"""

from __future__ import annotations

import argparse

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.training.gridworld_meta import GridWorldConfig, train_gridworld
from tools.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridWorld SNN 元学习训练。")
    parser.add_argument("--episodes", type=int, default=240, help="训练回合数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    parser.add_argument("--slip-prob", type=float, default=0.05, help="动作随机滑移概率")
    parser.add_argument("--step-cost", type=float, default=-0.01, help="每步惩罚")
    parser.add_argument("--goal-reward", type=float, default=1.0, help="终点奖励")
    parser.add_argument("--max-steps", type=int, default=60, help="单回合最大步数")
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
    train_gridworld(episodes=args.episodes, seed=args.seed, env_cfg=env_cfg)
    logger.info("GridWorld 元探索训练完成。")


if __name__ == "__main__":
    main()
