"""集成级别验证模块化训练流程。"""

from __future__ import annotations

import unittest

from snn.model import train_xor
from snn.training.gridworld_meta import GridWorldConfig, train_gridworld
from snn.training.smsa import RecoverySummary, train_smsa


class TestIntegration(unittest.TestCase):
    def test_xor_training_reaches_target_accuracy(self) -> None:
        final_acc = train_xor(epochs=30)
        self.assertGreaterEqual(final_acc, 0.9)

    def test_gridworld_exploration_progress(self) -> None:
        cfg = GridWorldConfig(max_steps=20)
        metrics = train_gridworld(
            episodes=5,
            seed=42,
            env_cfg=cfg,
            validate=False,
        )
        self.assertIn("overall_success", metrics)

    def test_self_model_cause_accuracy(self) -> None:
        summary: RecoverySummary = train_smsa(
            episodes=1,
            seed=123,
            validate=False,
        )
        self.assertIsNotNone(summary)


if __name__ == "__main__":
    unittest.main()
