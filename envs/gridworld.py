"""5x5 GridWorld 环境实现。"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


class GridWorld:
    """简易的 5x5 GridWorld。

    代理从 (0,0) 出发，通过上/下/左/右动作移动，终点是 (4,4)。
    每个动作有 0.1 概率随机滑移为任一动作，超出边界会停在原地。
    """

    _ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # 上
        1: (1, 0),  # 下
        2: (0, -1),  # 左
        3: (0, 1),  # 右
    }

    def __init__(
        self,
        *,
        size: int = 5,
        slip_prob: float = 0.1,
        step_cost: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        """初始化环境参数。"""
        if size <= 0:
            raise ValueError("size 必须为正整数。")
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError("slip_prob 必须在 [0,1] 范围内。")

        self.size = size
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)
        self._rng = random.Random(seed)

        self._position: Tuple[int, int] = (0, 0)
        self._steps: int = 0
        self._done: bool = False

    def reset(self) -> List[float]:
        """环境复位，返回当前位置的 one-hot 状态。"""
        self._position = (0, 0)
        self._steps = 0
        self._done = False
        return self._encode_position(self._position)

    def step(
        self, action: int
    ) -> Tuple[List[float], float, bool, Dict[str, object]]:
        """执行一步动作，返回 (obs, reward, done, info)。"""
        if self._done:
            raise RuntimeError("环境已终止，请先调用 reset()。")
        if action not in self._ACTION_DELTAS:
            raise ValueError("action 必须是 0~3 的整数。")

        applied_action = self._apply_slip(action)
        self._position = self._move(self._position, applied_action)
        self._steps += 1

        reward = self.step_cost
        goal_reached = self._position == self.goal

        if goal_reached:
            reward += self.goal_reward
            self._done = True
        elif self._steps >= self.max_steps:
            self._done = True

        obs = self._encode_position(self._position)
        info = {
            "position": self._position,
            "steps": self._steps,
            "goal_reached": goal_reached,
        }
        return obs, reward, self._done, info

    def _apply_slip(self, action: int) -> int:
        """以 slip_prob 的概率随机替换动作。"""
        if self._rng.random() < self.slip_prob:
            return self._rng.randrange(len(self._ACTION_DELTAS))
        return action

    def _move(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """根据动作移动代理，越界时保持在边界。"""
        dr, dc = self._ACTION_DELTAS[action]
        row = min(max(position[0] + dr, 0), self.size - 1)
        col = min(max(position[1] + dc, 0), self.size - 1)
        return row, col

    def _encode_position(self, position: Tuple[int, int]) -> List[float]:
        """将坐标编码为 one-hot 向量。"""
        length = self.size * self.size
        index = position[0] * self.size + position[1]
        obs = [0.0] * length
        obs[index] = 1.0
        return obs


__all__ = ["GridWorld"]
