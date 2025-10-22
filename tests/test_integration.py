"""集成级别验证脚本训练流程。"""

import unittest

from snn.model import train_xor
from snn_py_explore_auto import run_training as run_explore
from snn_self_agent import run_training as run_self_model


class TestIntegration(unittest.TestCase):
    def test_xor_training_reaches_target_accuracy(self) -> None:
        final_acc = train_xor(epochs=30, eval_replays=12)
        self.assertGreaterEqual(final_acc, 0.9)

    def test_gridworld_exploration_progress(self) -> None:
        run_explore(episodes=60)

    def test_self_model_cause_accuracy(self) -> None:
        run_self_model(episodes=90)


if __name__ == "__main__":
    unittest.main()
