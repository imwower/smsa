"""MetaLearner：用于自动调整尖峰神经网络的训练超参。"""

from __future__ import annotations

import ast
import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple


EvaluationFn = Callable[[Any, int], float]


class CodePatcher:
    """根据表达式动态构造替代导数并应用到 DenseLIF 层。"""

    DEFAULT_EXPRESSIONS = [
        "g / ((1.0 + g * v * v) ** 2)",
        "(1.0 - math.fabs(v) / (w if w > 1e-6 else 1e-6)) if math.fabs(v) <= w else 0.0",
        "1.0 / (1.0 + g * math.fabs(v))",
    ]

    def __init__(self, expressions: Sequence[str] | None = None) -> None:
        self.expressions = list(expressions) if expressions else list(self.DEFAULT_EXPRESSIONS)
        if not self.expressions:
            raise ValueError("必须提供至少一个代码表达式。")
        self.current_idx = -1
        self._validate_all()

    def _validate_all(self) -> None:
        for expr in self.expressions:
            self._validate(expr)

    def _validate(self, expr: str) -> None:
        tree = ast.parse(expr, mode="eval")
        allowed = {"v", "w", "g", "math"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise ValueError(f"表达式包含非法标识符: {node.id}")

    def _build(self, expr: str):
        namespace = {"math": math}
        code = (
            "def dynamic_surrogate(v, w=1.0, g=2.0):\n"
            f"    return {expr}\n"
        )
        exec(code, namespace, namespace)  # noqa: S102
        return namespace["dynamic_surrogate"]

    def next_expression(self) -> str:
        self.current_idx = (self.current_idx + 1) % len(self.expressions)
        return self.expressions[self.current_idx]

    def apply(self, layer: Any, expression: str | None = None) -> str:
        if expression is None:
            expression = self.next_expression()
        else:
            self._validate(expression)
        func = self._build(expression)
        if not hasattr(layer, "set_surrogate"):
            raise AttributeError("目标对象缺少 set_surrogate 方法。")
        layer.set_surrogate(func)
        return f"patch surrogate expr={expression}"

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
        ucb_c: float = 0.6,
        ab_episodes: int = 5,
        candidates: Sequence[str] | None = None,
    ) -> None:
        self.window = window
        self.min_delta = min_delta
        self.ucb_c = ucb_c
        self.ab_episodes = ab_episodes
        default_candidates = [
            "eta_up",
            "eta_down",
            "vth_up",
            "vth_down",
            "intrinsic_up",
            "intrinsic_down",
            "inner_up",
            "inner_down",
            "switch_surrogate",
            "patch_surrogate",
        ]
        self.candidates: List[str] = list(candidates) if candidates else default_candidates
        self.counts: Dict[str, int] = {c: 0 for c in self.candidates}
        self.totals: Dict[str, float] = {c: 0.0 for c in self.candidates}
        self.attempts = 0
        self._positives = 0
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
        exploration = sum(max(1, self.counts[c]) for c in self.candidates) + 1
        best_score = -float("inf")
        best_candidate = self.candidates[0]
        for action in self.candidates:
            trials = max(1, self.counts[action])
            mean = self.totals[action] / trials if self.counts[action] > 0 else 0.0
            bonus = math.sqrt(2.0 * math.log(exploration) / trials)
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
        return self._positives / float(self.attempts)

    def adapt(
        self,
        agent: Any,
        *,
        step: int,
        evaluate_fn: EvaluationFn,
    ) -> Tuple[Any, List[str]]:
        """执行一次自改尝试，返回可能更新的 agent 和日志。"""
        seed = self._seed_base + step * 97
        baseline_score = evaluate_fn(copy.deepcopy(agent), seed)
        action = self.select_candidate()
        original_snapshot = copy.deepcopy(agent)
        applied, info = agent.apply_modification(action)
        delta = -self.min_delta
        reverted = True
        messages: List[str] = []
        if applied:
            after_score = evaluate_fn(copy.deepcopy(agent), seed)
            delta = after_score - baseline_score
            if delta > 0.0:
                reverted = False
                self._positives += 1
            else:
                agent = original_snapshot
        else:
            agent = original_snapshot
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
        return agent, messages


__all__ = ["MetaLearner"]
