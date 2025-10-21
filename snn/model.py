"""最小化 XOR 任务的脉冲网络模型与训练工具。"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from typing import List, Sequence, Tuple

from .dense import DenseLIF
from .lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate


logger = logging.getLogger(__name__)


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def xor_dataset() -> List[Tuple[List[int], int]]:
    return [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]


def draw_poisson_spikes(bits: Sequence[int], high_rate: float, low_rate: float) -> List[int]:
    spikes = []
    for bit in bits:
        rate = high_rate if bit else low_rate
        spikes.append(1 if random.random() < rate else 0)
    return spikes


def cross_entropy_loss(probs: Sequence[float], label: int) -> float:
    return -math.log(max(probs[label], 1e-8))


def clip_value(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class SNNModel:
    """使用隐藏 LIF 层与线性读出的 XOR 网络。"""

    def __init__(
        self,
        n_in: int,
        hidden_size: int,
        n_classes: int,
        params: LIFParams,
        surrogate: str = "fast_sigmoid",
    ) -> None:
        surrogate_map = {
            "triangular": triangular_surrogate,
            "fast_sigmoid": fast_sigmoid_surrogate,
        }
        surrogate_fn = surrogate_map.get(surrogate, triangular_surrogate)
        self.hidden = DenseLIF(
            n_in=n_in,
            n_out=hidden_size,
            params=params,
            surrogate_fn=surrogate_fn,
        )
        self.n_classes = n_classes
        self.readout_weights = [
            [random.uniform(-0.3, 0.3) for _ in range(n_classes)]
            for _ in range(hidden_size)
        ]
        self.readout_bias = [0.0 for _ in range(n_classes)]

    def reset_state(self) -> None:
        self.hidden.reset_state()


def run_trial(
    model: SNNModel,
    bits: Sequence[int],
    label: int,
    steps: int,
    high_rate: float,
    low_rate: float,
    hidden_lr: float,
    readout_lr: float,
    clip: float,
) -> Tuple[bool, float]:
    model.reset_state()
    hidden_counts = [0 for _ in range(model.hidden.n_out)]
    eligibility_history: List[List[List[float]]] = []
    bias_history: List[List[float]] = []
    for _ in range(steps):
        pre_spikes = draw_poisson_spikes(bits, high_rate, low_rate)
        spikes, _, eligibility_snapshot, bias_snapshot = model.hidden.step(
            pre_spikes
        )
        hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]
        eligibility_history.append(copy.deepcopy(eligibility_snapshot))
        bias_history.append(bias_snapshot[:])
    rates = [count / float(steps) for count in hidden_counts]
    logits = []
    for c in range(model.n_classes):
        logit = model.readout_bias[c]
        for h in range(model.hidden.n_out):
            logit += model.readout_weights[h][c] * rates[h]
        logits.append(logit)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, label)
    grad_logits = [p for p in probs]
    grad_logits[label] -= 1.0
    learning_signals = []
    for h in range(model.hidden.n_out):
        signal = 0.0
        for c in range(model.n_classes):
            signal += (
                grad_logits[c] * model.readout_weights[h][c] / float(steps)
            )
        learning_signals.append(signal)
    for i in range(model.hidden.n_in):
        for h in range(model.hidden.n_out):
            grad = 0.0
            for elig in eligibility_history:
                grad += learning_signals[h] * elig[i][h]
            grad = clip_value(grad, clip)
            model.hidden.weights[i][h] -= hidden_lr * grad
    for h in range(model.hidden.n_out):
        grad = 0.0
        for bias_elig in bias_history:
            grad += learning_signals[h] * bias_elig[h]
        grad = clip_value(grad, clip)
        model.hidden.bias[h] -= hidden_lr * grad
    for h in range(model.hidden.n_out):
        for c in range(model.n_classes):
            grad = grad_logits[c] * rates[h]
            grad = clip_value(grad, clip)
            model.readout_weights[h][c] -= readout_lr * grad
    for c in range(model.n_classes):
        grad = clip_value(grad_logits[c], clip)
        model.readout_bias[c] -= readout_lr * grad
    prediction = 0 if probs[0] > probs[1] else 1
    return prediction == label, loss


def evaluate(
    model: SNNModel,
    steps: int,
    high_rate: float,
    low_rate: float,
    trials_per_example: int,
) -> float:
    dataset = xor_dataset()
    correct = 0
    total = 0
    for bits, label in dataset:
        for _ in range(trials_per_example):
            model.reset_state()
            hidden_counts = [0 for _ in range(model.hidden.n_out)]
            for _ in range(steps):
                pre_spikes = draw_poisson_spikes(bits, high_rate, low_rate)
                spikes, _, _, _ = model.hidden.step(pre_spikes)
                hidden_counts = [c + s for c, s in zip(hidden_counts, spikes)]
            rates = [count / float(steps) for count in hidden_counts]
            logits = []
            for c in range(model.n_classes):
                logit = model.readout_bias[c]
                for h in range(model.hidden.n_out):
                    logit += model.readout_weights[h][c] * rates[h]
                logits.append(logit)
            probs = softmax(logits)
            prediction = 0 if probs[0] > probs[1] else 1
            if prediction == label:
                correct += 1
            total += 1
    return correct / float(total)


def train_xor(
    epochs: int = 30,
    steps: int = 30,
    train_replays: int = 8,
    eval_replays: int = 20,
    surrogate: str = "fast_sigmoid",
    random_seed: int = 42,
) -> float:
    random.seed(random_seed)
    params = LIFParams(v_th=0.5, tau_m=8.0, tau_a=20.0, beta=0.5, refractory=2)
    model = SNNModel(
        n_in=2,
        hidden_size=4,
        n_classes=2,
        params=params,
        surrogate=surrogate,
    )
    hidden_lr = 0.3
    readout_lr = 0.5
    clip = 2.0
    high_rate = 0.95
    low_rate = 0.05
    dataset = xor_dataset()
    start = time.time()
    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        epoch_loss = 0.0
        epoch_hits = 0
        epoch_trials = 0
        for bits, label in dataset:
            for _ in range(train_replays):
                hit, loss = run_trial(
                    model,
                    bits,
                    label,
                    steps,
                    high_rate,
                    low_rate,
                    hidden_lr,
                    readout_lr,
                    clip,
                )
                epoch_hits += 1 if hit else 0
                epoch_trials += 1
                epoch_loss += loss
        eval_acc = evaluate(
            model,
            steps=steps,
            high_rate=high_rate,
            low_rate=low_rate,
            trials_per_example=eval_replays,
        )
        avg_loss = epoch_loss / float(max(epoch_trials, 1))
        train_acc = epoch_hits / float(max(epoch_trials, 1))
        logger.info(
            "第%02d轮 训练准确率 %.3f 验证准确率 %.3f 平均损失 %.3f",
            epoch,
            train_acc,
            eval_acc,
            avg_loss,
        )
        if eval_acc >= 0.9 and epoch >= 20:
            break
    duration = time.time() - start
    final_acc = evaluate(
        model,
        steps=steps,
        high_rate=high_rate,
        low_rate=low_rate,
        trials_per_example=eval_replays,
    )
    logger.info("训练耗时 %.2f 秒", duration)
    logger.info("最终评估准确率 %.3f", final_acc)
    return final_acc
