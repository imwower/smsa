"""策略头模块，实现基于 softmax 的 REINFORCE 更新。"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


class PolicyHead:
    """简单的全连接 softmax 策略头。"""

    def __init__(
        self,
        n_in: int,
        n_actions: int,
        lr: float,
        seed: int | None = None,
    ) -> None:
        if n_in <= 0 or n_actions <= 0:
            raise ValueError("n_in 和 n_actions 必须为正整数。")
        if lr <= 0:
            raise ValueError("学习率必须为正数。")

        self.n_in = n_in
        self.n_actions = n_actions
        self.lr = lr

        # 使用较小随机数初始化权重，便于训练收敛。
        init_scale = 0.1
        rng = random.Random(0)
        self.weights: List[List[float]] = [
            [rng.uniform(-init_scale, init_scale) for _ in range(n_in)]
            for _ in range(n_actions)
        ]
        self.bias: List[float] = [0.0 for _ in range(n_actions)]
        self._rng = random.Random(seed)

    def logits(self, counts: Sequence[float]) -> List[float]:
        """计算每个动作的线性得分。"""
        if len(counts) != self.n_in:
            raise ValueError("输入向量长度与 n_in 不匹配。")

        scores: List[float] = []
        for action in range(self.n_actions):
            # 计算 w·x + b
            w = self.weights[action]
            score = sum(w[i] * counts[i] for i in range(self.n_in)) + self.bias[action]
            scores.append(score)
        return scores

    def softmax(self, scores: Sequence[float]) -> List[float]:
        """对 logits 执行 softmax，返回概率分布。"""
        if len(scores) != self.n_actions:
            raise ValueError("scores 长度必须等于 n_actions。")

        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        denom = sum(exp_scores)
        if denom == 0.0:
            raise ZeroDivisionError("softmax 归一化因子为 0。")
        return [v / denom for v in exp_scores]

    def sample_action(self, probs: Sequence[float]) -> int:
        """按给定概率分布采样动作。"""
        if len(probs) != self.n_actions:
            raise ValueError("概率向量长度必须等于 n_actions。")
        total = sum(probs)
        if total <= 0.0:
            raise ValueError("概率向量总和必须为正值。")
        sample = self._rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob / total
            if sample <= cumulative or idx == self.n_actions - 1:
                return idx
        return self.n_actions - 1

    def policy_grad(self, probs: Sequence[float], action: int) -> List[float]:
        """计算策略梯度的动作维度项，即 p - 1_hot(a)。"""
        if len(probs) != self.n_actions:
            raise ValueError("概率向量长度必须等于 n_actions。")
        if action < 0 or action >= self.n_actions:
            raise ValueError("动作超出范围。")

        grad = [prob for prob in probs]
        grad[action] -= 1.0
        return grad

    def update(
        self,
        counts: Sequence[float],
        grad: Sequence[float],
        advantage: float,
    ) -> None:
        """根据 REINFORCE 规则更新权重和偏置。"""
        if len(counts) != self.n_in:
            raise ValueError("输入向量长度与 n_in 不匹配。")
        if len(grad) != self.n_actions:
            raise ValueError("梯度向量长度必须等于 n_actions。")

        for action in range(self.n_actions):
            g = grad[action] * advantage
            for i in range(self.n_in):
                self.weights[action][i] -= self.lr * g * counts[i]
            self.bias[action] -= self.lr * g


def _generate_toy_data(
    *, n_samples: int = 200, seed: int = 42
) -> List[Tuple[List[float], int]]:
    """生成两个线性可分的二维数据簇。"""
    if n_samples % 2 != 0:
        raise ValueError("n_samples 需要为偶数，以便平均分配类别。")

    rng = random.Random(seed)
    data: List[Tuple[List[float], int]] = []
    for _ in range(n_samples // 2):
        data.append(([rng.gauss(-1.0, 0.2), rng.gauss(-1.0, 0.2)], 0))
    for _ in range(n_samples // 2):
        data.append(([rng.gauss(1.0, 0.2), rng.gauss(1.0, 0.2)], 1))
    rng.shuffle(data)
    return data


def _train_toy_demo() -> None:
    """用 REINFORCE 训练策略头进行二分类，并打印 loss 曲线。"""
    head = PolicyHead(n_in=2, n_actions=2, lr=0.1)
    dataset = _generate_toy_data()

    losses: List[float] = []
    for step in range(200):
        counts, label = dataset[step % len(dataset)]
        scores = head.logits(counts)
        probs = head.softmax(scores)
        action = head.sample_action(probs)
        reward = 1.0 if action == label else 0.0
        advantage = reward - 0.5
        loss = -math.log(max(probs[label], 1e-12))
        losses.append(loss)

        grad = head.policy_grad(probs, action)
        head.update(counts, grad, advantage)

    # 评估分类准确率。
    correct = 0
    for counts, label in dataset:
        probs = head.softmax(head.logits(counts))
        pred = max(range(len(probs)), key=lambda idx: probs[idx])
        if pred == label:
            correct += 1
    accuracy = correct / len(dataset)

    print("Loss 曲线:", ", ".join(f"{l:.3f}" for l in losses[::20]))
    print(f"最终分类准确率: {accuracy:.2%}")


if __name__ == "__main__":
    _train_toy_demo()


__all__ = ["PolicyHead"]
