"""MetaLearner UCB 与回滚逻辑测试。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List

from meta.autoadapt import MetaLearner


@dataclass
class DummyAgent:
    current_modifier: str = "none"
    history: List[str] = field(default_factory=list)

    def apply_modification(self, candidate: str):
        self.current_modifier = candidate
        self.history.append(candidate)
        return True, None


class TestMetaLearner(unittest.TestCase):
    def test_ucb_prefers_higher_mean(self) -> None:
        learner = MetaLearner()
        learner.candidates = ["eta_up", "eta_down"]
        learner.counts = {c: 0 for c in learner.candidates}
        learner.totals = {c: 0.0 for c in learner.candidates}
        learner.counts["eta_up"] = 10
        learner.totals["eta_up"] = 5.0  # mean 0.5
        learner.counts["eta_down"] = 2
        learner.totals["eta_down"] = 0.2  # mean 0.1
        self.assertEqual(learner.select_candidate(), "eta_up")

    def test_adapt_with_fallback_and_revert(self) -> None:
        learner = MetaLearner()
        agent = DummyAgent()
        values = iter([1.0, 0.8, 1.1])

        def evaluate_fn(_agent: DummyAgent, seed: int) -> float:
            del seed
            return next(values)

        # 先手动设置 select_candidate 结果以触发回滚 + fallback。
        learner.candidates = ["eta_down", "eta_up"]
        learner.counts = {c: 0 for c in learner.candidates}
        learner.totals = {c: 0.0 for c in learner.candidates}

        import types

        learner.select_candidate = types.MethodType(lambda self: "eta_down", learner)

        updated_agent, messages = learner.adapt(
            agent,
            step=5,
            evaluate_fn=evaluate_fn,
        )

        self.assertEqual(updated_agent.current_modifier, "eta_up")
        self.assertEqual(len(messages), 2)
        self.assertIn("action=eta_down", messages[0])
        self.assertIn("reverted", messages[0])
        self.assertIn("action=eta_up", messages[1])
        self.assertIn("accepted", messages[1])
        self.assertEqual(learner.positives, 1)
        self.assertEqual(learner.attempts, 2)
        self.assertGreaterEqual(learner.positive_ratio(), 0.5)
        self.assertEqual(learner.counts["eta_down"], 1)
        self.assertEqual(learner.counts["eta_up"], 1)


if __name__ == "__main__":
    unittest.main()
