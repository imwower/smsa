"""MetaLearner UCB 与回滚逻辑测试。"""

from __future__ import annotations

import math
import types
import unittest
from dataclasses import dataclass
from typing import Iterator

from meta.autoadapt import MetaLearner


@dataclass
class DummyAgent:
    value: int = 0

    def apply_modification(self, candidate: str):
        del candidate
        self.value += 1
        return True, None


class TestMetaLearner(unittest.TestCase):
    def test_ucb_bonus_shrinks_with_trials(self) -> None:
        learner = MetaLearner()
        learner.candidates = ["alpha"]
        learner.counts = {"alpha": 1}
        learner.totals = {"alpha": 1.0}  # mean 1.0

        def compute_score() -> float:
            total = sum(max(1, learner.counts[c]) for c in learner.candidates)
            exploration = max(total, 2)
            mean = learner.totals["alpha"] / learner.counts["alpha"]
            bonus = learner.ucb_c * math.sqrt(
                2.0 * math.log(exploration) / learner.counts["alpha"]
            )
            return mean + bonus

        first_score = compute_score()
        learner.counts["alpha"] = 20
        learner.totals["alpha"] = 20.0  # 保持均值一致
        second_score = compute_score()

        self.assertLess(
            second_score,
            first_score,
            "试次数增加后 UCB 上界应逐渐收敛",
        )

    def test_adapt_rolls_back_on_negative_delta(self) -> None:
        learner = MetaLearner()
        learner.candidates = ["eta_up"]
        learner.counts = {"eta_up": 0}
        learner.totals = {"eta_up": 0.0}
        learner.select_candidate = types.MethodType(lambda self: "eta_up", learner)

        agent = DummyAgent()
        values: Iterator[float] = iter([1.0, 0.7])  # before, after

        def evaluate_fn(_agent: DummyAgent, seed: int) -> float:
            del seed
            return next(values)

        updated_agent, messages = learner.adapt(
            agent,
            step=3,
            evaluate_fn=evaluate_fn,
        )

        self.assertEqual(updated_agent.value, 0, "负增益应回滚参数改动")
        self.assertTrue(any("reverted" in msg for msg in messages))
        self.assertEqual(learner.counts["eta_up"], 1)
        self.assertAlmostEqual(learner.totals["eta_up"], -0.3)

    def test_adapt_keeps_positive_delta_and_updates_score(self) -> None:
        learner = MetaLearner()
        learner.candidates = ["eta_up"]
        learner.counts = {"eta_up": 0}
        learner.totals = {"eta_up": 0.0}
        learner.select_candidate = types.MethodType(lambda self: "eta_up", learner)

        agent = DummyAgent()
        values: Iterator[float] = iter([1.0, 1.2])  # before, after

        def evaluate_fn(_agent: DummyAgent, seed: int) -> float:
            del seed
            return next(values)

        updated_agent, messages = learner.adapt(
            agent,
            step=4,
            evaluate_fn=evaluate_fn,
        )

        self.assertEqual(updated_agent.value, 1, "正增益应保留参数改动")
        self.assertTrue(any("accepted" in msg for msg in messages))
        self.assertEqual(learner.counts["eta_up"], 1)
        self.assertAlmostEqual(learner.totals["eta_up"], 0.2)
        self.assertGreater(learner.positive_ratio(), 0.0)


if __name__ == "__main__":
    unittest.main()
