"""自适应任务调度器：结合 UCB1 与 ε-贪心策略，为 SMSA 各子系统分配资源。"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Tuple


__all__ = ["Scheduler", "TaskSelection", "SchedulerConfig"]


class TaskSelection(str):
    """任务选择结果，继承自字符串并附带推荐预算。"""

    def __new__(cls, task: str, budget_value: int, budget_unit: str) -> "TaskSelection":
        obj = str.__new__(cls, task)
        obj.budget = budget_value
        obj.unit = budget_unit
        return obj

    def __iter__(self):
        yield str(self)
        yield (self.budget, self.unit)


@dataclass
class SchedulerConfig:
    window: int = 10
    epsilon: float = 0.1
    ucb_c: float = 0.8
    csv_path: Path = Path("runs/scheduler.csv")
    state_path: Path = Path("runs/scheduler_state.json")


class Scheduler:
    """根据近端收益窗口动态协调 RL/LM/自适应与对外发布任务。"""

    TASKS: Tuple[str, ...] = ("rl", "lm", "autoadapt", "post")
    BUDGETS: Dict[str, Tuple[int, str]] = {
        "rl": (40, "episodes"),
        "lm": (800, "lines"),
        "autoadapt": (6, "steps"),
        "post": (220, "words"),
    }
    CSV_FIELDNAMES: Tuple[str, ...] = (
        "timestamp",
        "step",
        "task",
        "raw_reward",
        "energy_penalty",
        "penalized_reward",
        "window_mean",
        "window_std",
        "normalized_reward",
        "count",
        "budget_value",
        "budget_unit",
        "note",
    )

    def __init__(self, *, config: SchedulerConfig | None = None, seed: int | None = None) -> None:
        """初始化调度器，创建奖励缓存并尝试恢复历史状态。"""
        self.config = config or SchedulerConfig()
        self.buffers: Dict[str, Deque[float]] = {}
        self.counts: Dict[str, int] = {}
        self.total_updates = 0
        self.step = 0
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)
        else:
            self._rng.seed()
        for task in self.TASKS:
            self._register_task(task)
        self._ensure_csv()
        self.load_state()

    def _ensure_csv(self) -> None:
        path = self.config.csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDNAMES)
                writer.writeheader()

    def select_next(self, epsilon: float | None = None) -> TaskSelection:
        """按照 ε-贪心 + UCB1 得分挑选下一项任务并返回推荐预算。"""
        epsilon = self.config.epsilon if epsilon is None else epsilon
        for task in self.TASKS:
            if self.counts.get(task, 0) == 0:
                return self._make_selection(task)
        if self._rng.random() < epsilon:
            task = self._rng.choice(self.TASKS)
            return self._make_selection(task)
        scores = {task: self._ucb_score(task) for task in self.TASKS}
        task = max(scores.items(), key=lambda item: item[1])[0]
        return self._make_selection(task)

    def _make_selection(self, task: str) -> TaskSelection:
        budget_value, budget_unit = self.BUDGETS.get(task, (1, "units"))
        return TaskSelection(task, budget_value, budget_unit)

    def _ucb_score(self, task: str) -> float:
        mean = self._window_mean(task)
        count = max(1, self.counts.get(task, 0))
        total = max(1, self.total_updates)
        exploration = math.sqrt(2.0 * math.log(total + 1.0) / count)
        return mean + self.config.ucb_c * exploration

    def update(self, task: str, reward: float, *, energy_penalty: float = 0.0, note: str = "") -> None:
        """记录指定任务的收益条目，扣除能耗并写入日志。"""
        if task not in self.buffers:
            self._register_task(task)
        penalized = reward - energy_penalty
        buffer = self.buffers[task]
        buffer.append(penalized)
        self.counts[task] = self.counts.get(task, 0) + 1
        self.total_updates += 1
        window_mean = self._window_mean(task)
        window_std = self._window_std(task)
        normalized = 0.0
        if window_std > 0.0:
            normalized = (penalized - window_mean) / window_std
        self.step += 1
        self._write_csv(
            task=task,
            raw_reward=reward,
            energy_penalty=energy_penalty,
            penalized_reward=penalized,
            window_mean=window_mean,
            window_std=window_std,
            normalized_reward=normalized,
            count=self.counts[task],
            note=note,
        )
        self.save_state()

    def _write_csv(
        self,
        *,
        task: str,
        raw_reward: float,
        energy_penalty: float,
        penalized_reward: float,
        window_mean: float,
        window_std: float,
        normalized_reward: float,
        count: int,
        note: str,
    ) -> None:
        budget_value, budget_unit = self.BUDGETS.get(task, (1, "units"))
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": self.step,
            "task": task,
            "raw_reward": f"{raw_reward:.6f}",
            "energy_penalty": f"{energy_penalty:.6f}",
            "penalized_reward": f"{penalized_reward:.6f}",
            "window_mean": f"{window_mean:.6f}",
            "window_std": f"{window_std:.6f}",
            "normalized_reward": f"{normalized_reward:.6f}",
            "count": count,
            "budget_value": budget_value,
            "budget_unit": budget_unit,
            "note": note,
        }
        with self.config.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDNAMES)
            writer.writerow(row)

    def save_state(self) -> None:
        """序列化内部状态，支持断点续跑。"""
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "buffers": {task: list(buffer) for task, buffer in self.buffers.items()},
            "counts": self.counts,
            "total_updates": self.total_updates,
            "step": self.step,
            "random_state": self._serialize_random_state(self._rng.getstate()),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        """尝试从 JSON 状态文件恢复；若无文件则返回 False。"""
        path = self.config.state_path
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        buffers = state.get("buffers", {})
        counts = state.get("counts", {})
        for task, values in buffers.items():
            self.buffers[task] = deque(
                (float(v) for v in values[-self.config.window :]),
                maxlen=self.config.window,
            )
        for task in self.TASKS:
            if task not in self.buffers:
                self.buffers[task] = deque(maxlen=self.config.window)
        for task, count in counts.items():
            self.counts[task] = int(count)
        for task in self.TASKS:
            self.counts.setdefault(task, 0)
        self.total_updates = int(state.get("total_updates", self.total_updates))
        self.step = int(state.get("step", self.step))
        random_state = state.get("random_state")
        if random_state is not None:
            self._rng.setstate(self._deserialize_random_state(random_state))
        return True

    def _register_task(self, task: str) -> None:
        if task not in self.buffers:
            self.buffers[task] = deque(maxlen=self.config.window)
        if task not in self.counts:
            self.counts[task] = 0

    def _window_mean(self, task: str) -> float:
        buffer = self.buffers[task]
        if not buffer:
            return 0.0
        return sum(buffer) / float(len(buffer))

    def _window_std(self, task: str) -> float:
        buffer = self.buffers[task]
        if len(buffer) < 2:
            return 0.0
        mean = self._window_mean(task)
        variance = sum((value - mean) ** 2 for value in buffer) / float(len(buffer))
        return math.sqrt(max(variance, 0.0))

    def describe(self) -> Dict[str, Dict[str, float]]:
        """输出各任务近期统计信息，便于调试观察。"""
        summary: Dict[str, Dict[str, float]] = {}
        for task in self.TASKS:
            summary[task] = {
                "count": self.counts[task],
                "mean": self._window_mean(task),
                "std": self._window_std(task),
            }
        return summary

    @staticmethod
    def _serialize_random_state(state: Any) -> Any:
        if isinstance(state, tuple):
            return [Scheduler._serialize_random_state(item) for item in state]
        return state

    @staticmethod
    def _deserialize_random_state(state: Any) -> Any:
        if isinstance(state, list):
            return tuple(Scheduler._deserialize_random_state(item) for item in state)
        return state
