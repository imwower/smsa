"""PolicyHead 单元测试。"""

from __future__ import annotations

import math
import random
import unittest

from snn.policy import PolicyHead


class PolicyHeadTest(unittest.TestCase):
    """验证策略头在简单线性可分任务上的训练效果。"""

    @staticmethod
    def _make_dataset(n_samples: int = 200, seed: int = 7):
        if n_samples % 2 != 0:
            raise ValueError("n_samples 需为偶数。")

        rng = random.Random(seed)
        data = []
        for _ in range(n_samples // 2):
            data.append(([rng.gauss(-1.0, 0.15), rng.gauss(-1.0, 0.15)], 0))
        for _ in range(n_samples // 2):
            data.append(([rng.gauss(1.0, 0.15), rng.gauss(1.0, 0.15)], 1))
        rng.shuffle(data)
        return data

    def test_reinforce_trains_linear_task(self) -> None:
        dataset = self._make_dataset()
        head = PolicyHead(n_in=2, n_actions=2, lr=0.15, seed=0)

        baseline = 0.5
        baseline_beta = 0.05
        losses: list[float] = []
        for step in range(200):
            features, label = dataset[step % len(dataset)]
            probs = head.softmax(head.logits(features))
            action = head.sample_action(probs)
            reward = 1.0 if action == label else 0.0
            advantage = reward - baseline
            grad = head.policy_grad(probs, action)
            head.update(features, grad, advantage)
            baseline += baseline_beta * (reward - baseline)
            losses.append(-math.log(max(probs[label], 1e-9)))

        correct = 0
        for features, label in dataset:
            probs = head.softmax(head.logits(features))
            pred = max(range(len(probs)), key=lambda idx: probs[idx])
            if pred == label:
                correct += 1

        accuracy = correct / len(dataset)
        headroom = len(losses) // 4 or 1
        early = sum(losses[:headroom]) / headroom
        late = sum(losses[-headroom:]) / headroom
        self.assertGreaterEqual(accuracy, 0.9, "策略头应在 200 步内学会线性任务")
        self.assertLess(
            late,
            early,
            f"损失应下降，初期 {early:.3f} 晚期 {late:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
