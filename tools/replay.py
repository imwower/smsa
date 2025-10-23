"""Simple replay buffers backed by deque."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Sequence, Tuple


class ReplayBuffer:
    """Fixed-size replay buffer for SMSA dream and recovery phases."""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self.data: Deque[Tuple[int, int, List[int], float, int]] = deque(maxlen=capacity)

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


__all__ = ["ReplayBuffer"]
