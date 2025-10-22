"""MetaLearner UCB 与回滚逻辑测试。"""

import unittest
from dataclasses import dataclass, field
from typing import List, Tuple
from unittest.mock import patch

from snn_py_explore_auto import GridWorldConfig, MetaLearner


@dataclass
class DummyAgent:
    current_modifier: str = "none"
    history: List[str] = field(default_factory=list)

    def apply_modification(self, candidate: str) -> Tuple[bool, None]:
        self.current_modifier = candidate
        self.history.append(candidate)
        return True, None


class TestMetaLearner(unittest.TestCase):
    def test_ucb_selects_better_candidate(self) -> None:
        ml = MetaLearner()
        ml.candidates = ["alpha", "beta"]
        ml.counts["alpha"] = 10
        ml.totals["alpha"] = 5.0  # mean 0.5
        ml.counts["beta"] = 2
        ml.totals["beta"] = 0.1  # mean 0.05
        choice = ml.select_candidate()
        self.assertEqual(choice, "alpha")

    @patch("snn_py_explore_auto.evaluate_agent", side_effect=[1.0, 0.8, 1.15])
    @patch.object(MetaLearner, "select_candidate", return_value="eta_down")
    def test_adapt_rolls_back_and_keeps_positive(self, select_mock, eval_mock) -> None:
        ml = MetaLearner()
        agent = DummyAgent()
        env_cfg = GridWorldConfig()

        updated_agent, message = ml.adapt(agent, env_cfg, episode=5, history=[0.1] * 20)

        self.assertIn("action=eta_up", message)
        self.assertEqual(updated_agent.current_modifier, "eta_up")
        self.assertEqual(ml.counts["eta_down"], 1)
        self.assertEqual(ml.counts["eta_up"], 1)
        self.assertEqual(ml.positives, 1)
        self.assertGreater(ml.positive_ratio(), 0.0)


if __name__ == "__main__":
    unittest.main()
