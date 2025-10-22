"""MetaLearner：用于自动调整尖峰神经网络的训练超参。"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple


EvaluationFn = Callable[[Any, int], float]

@dataclass
class AdaptationLog:
    """记录一次候选改动的结果。"""

    step: int
    action: str
    delta: float
    reverted: bool
    info: str | None
    positive_ratio: float

    def format(self) -> str:
        status = "reverted" if self.reverted else "accepted"
        suffix = f", info={self.info}" if self.info else ""
        return (
            f"[meta] step {self.step:03d} action={self.action} "
            f"delta={self.delta:.3f} ({status}, pos_ratio={self.positive_ratio:.2f}"
            f"{suffix})"
        )


class MetaLearner:
    """使用 UCB 选择自改动作，并通过 A/B 测试验证。"""

    def __init__(
        self,
        *,
        window: int = 20,
        min_delta: float = 0.05,
        ucb_c: float = 0.4,
        ab_episodes: int = 5,
        fallback_action: str = "eta_up",
    ) -> None:
        self.window = window
        self.min_delta = min_delta
        self.ucb_c = ucb_c
        self.ab_episodes = ab_episodes
        self.fallback_action = fallback_action
        self.candidates: List[str] = [
            "eta_up",
            "eta_down",
            "vth_up",
            "vth_down",
            "intrinsic_up",
            "intrinsic_down",
            "inner_up",
            "inner_down",
            "add_neuron",
            "prune_neuron",
            "switch_surrogate",
            "patch_surrogate",
        ]
        self.counts: Dict[str, int] = {c: 0 for c in self.candidates}
        self.totals: Dict[str, float] = {c: 0.0 for c in self.candidates}
        self.attempts = 0
        self.positives = 0
        self._seed_base = 1337
        self._rng = random.Random(2024)

    def should_trigger(self, history: Sequence[float]) -> bool:
        """判断是否进入平台期。"""
        if len(history) < self.window:
            return False
        window_values = history[-self.window :]
        return (max(window_values) - min(window_values)) < self.min_delta

    def select_candidate(self) -> str:
        """使用 UCB 对候选改动进行选择。"""
        total = sum(max(1, self.counts[c]) for c in self.candidates)
        exploration = max(total, 2)
        best_score = -float("inf")
        best_candidate = self.candidates[0]
        for action in self.candidates:
            mean = (
                self.totals[action] / self.counts[action]
                if self.counts[action] > 0
                else 0.05
            )
            bonus = math.sqrt(
                2.0 * math.log(exploration) / max(1, self.counts[action])
            )
            score = mean + self.ucb_c * bonus
            if (
                score > best_score + 1e-9
                or (abs(score - best_score) <= 1e-9 and self._rng.random() < 0.5)
            ):
                best_score = score
                best_candidate = action
        return best_candidate

    def positive_ratio(self) -> float:
        """返回带来正收益的改动比例。"""
        if self.attempts == 0:
            return 1.0
        return self.positives / float(self.attempts)

    def adapt(
        self,
        agent: Any,
        *,
        step: int,
        evaluate_fn: EvaluationFn,
    ) -> Tuple[Any, List[str]]:
        """执行一次自改尝试，返回可能更新的 agent 和日志。"""
        seed = self._seed_base + step * 97
        before = evaluate_fn(copy.deepcopy(agent), seed)
        candidates = [self.select_candidate()]
        if self.fallback_action not in candidates:
            candidates.append(self.fallback_action)

        messages: List[str] = []
        for idx, action in enumerate(candidates):
            backup = copy.deepcopy(agent)
            applied, info = agent.apply_modification(action)
            delta = -self.min_delta
            reverted = True
            if applied:
                after = evaluate_fn(copy.deepcopy(agent), seed)
                delta = after - before
                if delta >= 0.0:
                    reverted = False
                    self.positives += 1
                else:
                    agent = backup
            else:
                agent = backup
            self.attempts += 1
            self.counts[action] += 1
            self.totals[action] += delta
            log = AdaptationLog(
                step=step,
                action=action,
                delta=delta,
                reverted=reverted,
                info=info,
                positive_ratio=self.positive_ratio(),
            )
            message = log.format()
            messages.append(message)
            print(message)
            if not reverted:
                break
        return agent, messages


__all__ = ["MetaLearner"]
