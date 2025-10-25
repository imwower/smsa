"""MetaLearner：用于自动调整尖峰神经网络的训练超参。"""

from __future__ import annotations

import ast
import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Set


EvaluationFn = Callable[[Any, int], float]


class CodePatcher:
    """根据表达式动态构造替代导数并应用到 DenseLIF 层。"""

    DEFAULT_EXPRESSIONS = [
        "g / ((1.0 + g * v * v) ** 2)",
        "(1.0 - math.fabs(v) / (w if w > 1e-6 else 1e-6)) if math.fabs(v) <= w else 0.0",
        "1.0 / (1.0 + g * math.fabs(v))",
    ]

    def __init__(self, expressions: Sequence[str] | None = None) -> None:
        self.expressions = (
            list(expressions) if expressions else list(self.DEFAULT_EXPRESSIONS)
        )
        if not self.expressions:
            raise ValueError("必须提供至少一个替代导数表达式。")
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

    def apply(self, target: Any, expression: str | None = None) -> str:
        if expression is None:
            expression = self.next_expression()
        else:
            self._validate(expression)
        func = self._build(expression)
        patched = self._patch_recursive(target, func, set())
        if patched == 0:
            raise AttributeError("未找到可替换 surrogate 的 LIF 层。")
        return f"patch surrogate expr={expression} layers={patched}"

    def _patch_recursive(
        self,
        target: Any,
        func: Callable[[float], float],
        visited: Set[int],
    ) -> int:
        if target is None:
            return 0
        obj_id = id(target)
        if obj_id in visited:
            return 0
        visited.add(obj_id)
        patched = 0
        set_surrogate = getattr(target, "set_surrogate", None)
        if callable(set_surrogate):
            set_surrogate(func)
            patched += 1
        if isinstance(target, dict):
            for value in target.values():
                patched += self._patch_recursive(value, func, visited)
        elif isinstance(target, (list, tuple, set)):
            for item in target:
                patched += self._patch_recursive(item, func, visited)
        elif hasattr(target, "__dict__"):
            for value in target.__dict__.values():
                patched += self._patch_recursive(value, func, visited)
        return patched


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
        suffix = f" info={self.info}" if self.info else ""
        return (
            f"[meta] ep{self.step:03d} action={self.action} "
            f"delta={self.delta:.3f} ({status}, pos={self.positive_ratio:.2f}){suffix}"
        )


class MetaLearner:
    """使用 UCB 选择自改动作，并通过 A/B 测试验证。"""

    DEFAULT_ACTIONS = [
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

    def __init__(
        self,
        *,
        plateau_window: int = 20,
        plateau_min_delta: float = 0.05,
        window: int | None = None,
        min_delta: float | None = None,
        ucb_c: float = 0.6,
        ab_rounds: int = 5,
        ab_episodes: int | None = None,
        actions: Sequence[str] | None = None,
        candidates: Sequence[str] | None = None,
        fallback_action: str | None = None,
        seed: int | None = None,
    ) -> None:
        if window is not None:
            plateau_window = window
        if min_delta is not None:
            plateau_min_delta = min_delta
        if ab_episodes is not None:
            ab_rounds = ab_episodes
        if plateau_window <= 0:
            raise ValueError("plateau_window 需为正整数。")
        if plateau_min_delta < 0:
            raise ValueError("plateau_min_delta 需为非负数。")
        if ab_rounds <= 0:
            raise ValueError("ab_rounds 需为正整数。")

        self.window = plateau_window
        self.min_delta = plateau_min_delta
        self.ucb_c = ucb_c
        self.ab_rounds = ab_rounds
        self.ab_episodes = ab_rounds
        action_list: List[str] = []
        if actions is not None:
            action_list = list(actions)
        elif candidates is not None:
            action_list = list(candidates)
        else:
            action_list = list(self.DEFAULT_ACTIONS)
        self.fallback_action = fallback_action
        if self.fallback_action and self.fallback_action not in action_list:
            action_list.append(self.fallback_action)
        self.actions = action_list
        if not self.actions:
            raise ValueError("必须提供至少一个自改动作。")

        self.counts: Dict[str, int] = {action: 0 for action in self.actions}
        self.totals: Dict[str, float] = {action: 0.0 for action in self.actions}
        self.attempts = 0
        self._positives = 0
        self._seed_base = 2027
        self._rng = random.Random(seed)

    def should_trigger(self, history: Sequence[float]) -> bool:
        """判断最近窗口是否进入平台期。"""
        if len(history) < self.window:
            return False
        recent = history[-self.window :]
        return (max(recent) - min(recent)) < self.min_delta

    def positive_ratio(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self._positives / float(self.attempts)

    def select_candidate(self) -> str:
        """根据 UCB 估计选择一个动作。"""
        total_trials = sum(max(1, self.counts[a]) for a in self.actions) + 1
        best_action = self.actions[0]
        best_score = -float("inf")
        for action in self.actions:
            trials = self.counts[action]
            mean = self.totals[action] / trials if trials > 0 else 0.0
            bonus = math.sqrt(
                2.0 * math.log(total_trials) / max(1, trials)
            )
            score = mean + self.ucb_c * bonus
            if (
                score > best_score + 1e-9
                or (abs(score - best_score) <= 1e-9 and self._rng.random() < 0.5)
            ):
                best_score = score
                best_action = action
        return best_action

    def _evaluate_agent(
        self,
        agent: Any,
        evaluate_fn: EvaluationFn,
        base_seed: int,
    ) -> List[float]:
        scores: List[float] = []
        for offset in range(self.ab_rounds):
            seed = base_seed + offset
            score = evaluate_fn(copy.deepcopy(agent), seed)
            scores.append(score)
        return scores

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    def adapt(
        self,
        agent: Any,
        *,
        step: int,
        evaluate_fn: EvaluationFn,
    ) -> Tuple[Any, List[str]]:
        """执行一次自改尝试，返回改动后的 agent 及日志。"""
        if not callable(evaluate_fn):
            raise TypeError("evaluate_fn 必须为可调用对象。")

        base_seed = self._seed_base + step * 97
        baseline_scores = self._evaluate_agent(agent, evaluate_fn, base_seed)
        primary = self.select_candidate()
        actions_to_try = [primary]
        if (
            self.fallback_action
            and self.fallback_action not in actions_to_try
        ):
            actions_to_try.append(self.fallback_action)

        messages: List[str] = []
        chosen_agent = agent
        accepted = False
        for action in actions_to_try:
            if action not in self.counts:
                self.counts[action] = 0
                self.totals[action] = 0.0

            trial_agent = copy.deepcopy(agent)
            applied = False
            info: str | None = None
            if hasattr(trial_agent, "apply_modification"):
                applied, info = trial_agent.apply_modification(action)
            else:
                info = "missing apply_modification"

            delta = 0.0
            reverted = True
            if applied:
                candidate_scores = self._evaluate_agent(
                    trial_agent,
                    evaluate_fn,
                    base_seed,
                )
                delta = (
                    self._mean(candidate_scores) - self._mean(baseline_scores)
                )
                self.counts[action] += 1
                self.totals[action] += delta
                self.attempts += 1
                if delta >= 0.0:
                    reverted = False
                    chosen_agent = trial_agent
                    accepted = True
                    if delta > 0.0:
                        self._positives += 1
                else:
                    info = info or "negative delta"
            else:
                info = info or "apply failed"
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
            if accepted:
                agent = chosen_agent
                break

        return agent, messages


__all__ = ["MetaLearner", "CodePatcher"]
