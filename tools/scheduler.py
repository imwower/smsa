"""Simple task scheduler mixing UCB1 and ε-greedy for SMSA workflows."""

from __future__ import annotations

import csv
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple


__all__ = ["Scheduler", "TaskSelection"]


class TaskSelection(str):
    """String-like selection result that also carries a suggested budget."""

    def __new__(cls, task: str, budget_value: int, budget_unit: str) -> "TaskSelection":
        obj = str.__new__(cls, task)
        obj.budget = budget_value
        obj.unit = budget_unit
        return obj


@dataclass
class SchedulerConfig:
    window: int = 10
    epsilon: float = 0.1
    ucb_c: float = 0.8
    csv_path: Path = Path("runs/scheduler.csv")


class Scheduler:
    """Coordinate RL/LM/AutoAdapt jobs based on recent normalized rewards."""

    TASKS: Tuple[str, ...] = ("rl", "lm", "autoadapt")
    BUDGETS: Dict[str, Tuple[int, str]] = {
        "rl": (40, "episodes"),
        "lm": (800, "lines"),
        "autoadapt": (6, "steps"),
    }

    def __init__(self, *, config: SchedulerConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SchedulerConfig()
        self.buffers: Dict[str, Deque[float]] = {
            task: deque(maxlen=self.config.window) for task in self.TASKS
        }
        self.counts: Dict[str, int] = {task: 0 for task in self.TASKS}
        self.total_updates = 0
        self._rng = random.Random(seed)
        self._ensure_csv()
        self.step = 0

    def _ensure_csv(self) -> None:
        path = self.config.csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()

    def select_next(self) -> TaskSelection:
        """Pick the next task using ε-greedy + UCB1 bonus."""
        for task in self.TASKS:
            if self.counts[task] == 0:
                return self._make_selection(task)
        if self._rng.random() < self.config.epsilon:
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
        count = max(1, self.counts[task])
        total = max(1, self.total_updates)
        exploration = math.sqrt(2.0 * math.log(total + 1.0) / count)
        return mean + self.config.ucb_c * exploration

    def update(self, task: str, reward: float, *, energy_penalty: float = 0.0) -> None:
        """Record the observed reward (minus energy) and log to CSV."""
        if task not in self.buffers:
            raise KeyError(f"Unknown task: {task}")
        penalized = reward - energy_penalty
        buffer = self.buffers[task]
        buffer.append(penalized)
        self.counts[task] += 1
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
        )

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
        }
        with self.config.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writerow(row)

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
        """Return current statistics for inspection."""
        summary: Dict[str, Dict[str, float]] = {}
        for task in self.TASKS:
            summary[task] = {
                "count": self.counts[task],
                "mean": self._window_mean(task),
                "std": self._window_std(task),
            }
        return summary
