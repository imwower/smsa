"""GridWorld 随机策略冒烟测试。"""

from __future__ import annotations

import pathlib
import random
import sys

# 将项目根目录加入 sys.path，便于脚本直接运行。
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.gridworld import GridWorld


def run_random_policy(episodes: int = 200) -> None:
    """使用随机策略运行若干回合，并打印统计。"""
    env = GridWorld()
    policy_rng = random.Random()

    total_steps = 0
    successes = 0

    for _ in range(episodes):
        env.reset()
        done = False

        while not done:
            action = policy_rng.randrange(4)
            _, _, done, info = env.step(action)
            total_steps += 1
            if done and info["goal_reached"]:
                successes += 1

    avg_steps = total_steps / episodes
    success_rate = successes / episodes
    print(f"平均步长/到达率: {avg_steps:.2f}/{success_rate:.2%}")


if __name__ == "__main__":
    run_random_policy()
