"""Training loop that lets the SelfModel learn a tiny Chinese corpus autonomously."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from meta.autoadapt import MetaLearner
from snn.datasets import (
    CharVocab,
    MiniCorpusRecord,
    iter_char_sequences,
    build_char_vocab,
    load_minicorpus,
)
from snn.lif import LIFParams, fast_sigmoid_surrogate, triangular_surrogate
from snn.selfmodel import SelfModel
from tools.logger import get_logger


logger = get_logger(__name__)

DOMAIN_TO_ACTION = {"生活": 0, "职场": 1, "出行": 2, "学习": 3}
URGENCY_TO_VALUE = {"低": 0.25, "中": 0.55, "高": 0.85}
TONE_TO_VALUE = {"积极": 0.9, "中性": 0.65, "谨慎": 0.4}
SUPPORTED_ACTIONS = [
    "eta_up",
    "eta_down",
    "vth_up",
    "vth_down",
    "inner_up",
    "inner_down",
    "switch_surrogate",
    "patch_surrogate",
]


@dataclass
class EpisodeMetrics:
    avg_nll: float
    cause_acc: float
    energy_mse: float
    mean_spike_norm: float


@dataclass
class MiniCorpusResult:
    tail_nll: float
    tail_cause_acc: float
    tail_energy_mse: float
    tail_spike_norm: float
    generated_samples: List[str]
    positive_ratio: float
    history: List[EpisodeMetrics]


@dataclass
class MiniCorpusConfig:
    dataset_path: str = "data/minicorpus.jsonl"
    train_limit: int = 2200
    val_limit: int = 600
    vocab_size: int = 384
    episodes: int = 40
    seed: int | None = 0
    meta_window: int = 8
    meta_min_delta: float = 0.01
    log_interval: int = 200


class MiniCorpusTrainer:
    """Wrap SelfModel training utilities for the minimal Chinese corpus."""

    def __init__(self, config: MiniCorpusConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        records = load_minicorpus(config.dataset_path, seed=config.seed)
        if len(records) < 20:
            raise ValueError("语料不足，至少需要 20 条记录。")
        self.vocab = build_char_vocab(records, max_size=config.vocab_size)
        shuffled = list(records)
        self.rng.shuffle(shuffled)
        train_cut = min(len(shuffled), config.train_limit)
        val_cut = min(len(shuffled) - train_cut, config.val_limit)
        train_records = shuffled[:train_cut]
        val_records = shuffled[train_cut : train_cut + val_cut]
        if not val_records:
            val_records = shuffled[train_cut:]
        self.train_sequences = list(iter_char_sequences(train_records, self.vocab))
        self.val_sequences = list(iter_char_sequences(val_records, self.vocab))
        lif_params = LIFParams(v_th=0.5, tau_m=9.5, tau_a=18.0, beta=0.42, refractory=2)
        self.self_model = SelfModel(
            obs_dim=self.vocab.size,
            action_dim=len(DOMAIN_TO_ACTION),
            hidden_size=48,
            lif_params=lif_params,
            inner_steps=14,
            hidden_lr=0.09,
            readout_lr=0.18,
            reward_weight=0.4,
            energy_weight=0.3,
            cause_weight=0.5,
        )
        self.surrogate_name = "fast_sigmoid"

    @property
    def action_dim(self) -> int:
        return len(DOMAIN_TO_ACTION)

    def train_epoch(self) -> EpisodeMetrics:
        self.rng.shuffle(self.train_sequences)
        total_nll = 0.0
        total_cause = 0.0
        total_energy = 0.0
        total_spike = 0.0
        steps = 0
        total_sentences = len(self.train_sequences)
        for idx, (record, sequence) in enumerate(self.train_sequences, 1):
            total_nll, total_cause, total_energy, total_spike, steps = self._train_sentence(
                record,
                sequence,
                total_nll,
                total_cause,
                total_energy,
                total_spike,
                steps,
            )
            if (
                self.config.log_interval > 0
                and idx % self.config.log_interval == 0
            ):
                running_nll = total_nll / float(max(steps, 1))
                running_cause = total_cause / float(max(steps, 1))
                running_energy = total_energy / float(max(steps, 1))
                logger.info(
                    "[progress] sentences %d/%d avg_nll=%.3f cause=%.2f energy=%.3f",
                    idx,
                    total_sentences,
                    running_nll,
                    running_cause,
                    running_energy,
                )
        avg_nll = total_nll / float(max(steps, 1))
        cause_acc = total_cause / float(max(steps, 1))
        energy_mse = total_energy / float(max(steps, 1))
        mean_spike = total_spike / float(max(steps, 1))
        return EpisodeMetrics(
            avg_nll=avg_nll,
            cause_acc=cause_acc,
            energy_mse=energy_mse,
            mean_spike_norm=mean_spike,
        )

    def _train_sentence(
        self,
        record: MiniCorpusRecord,
        sequence: Sequence[int],
        total_nll: float,
        total_cause: float,
        total_energy: float,
        total_spike: float,
        steps: int,
    ) -> Tuple[float, float, float, float, int]:
        unique_chars: set[int] = set()
        action_idx = DOMAIN_TO_ACTION.get(record.domain, 0)
        urgency_val = URGENCY_TO_VALUE.get(record.urgency, 0.55)
        tone_val = TONE_TO_VALUE.get(record.tone, 0.65)
        seq_len = max(len(sequence) - 1, 1)
        for pos in range(len(sequence) - 1):
            current_id = sequence[pos]
            next_id = sequence[pos + 1]
            unique_chars.add(current_id)
            position_ratio = pos / float(seq_len)
            unique_ratio = len(unique_chars) / float(max(len(record.text), 1))
            extras = self._extra_features(position_ratio, unique_ratio, urgency_val, tone_val)
            features = self._assemble_features(current_id, action_idx, extras)
            state = self.self_model.forward(features)
            reward_target, energy_target = self._targets(position_ratio, tone_val, urgency_val, unique_ratio)
            cause_label = 1 if record.tone == "积极" else 0
            self.self_model.update(
                state=state,
                next_obs_index=next_id,
                reward_target=reward_target,
                energy_target=energy_target,
                cause_label=cause_label,
            )
            prob_next = max(state.probs_next[next_id], 1e-8)
            total_nll += -math.log(prob_next)
            cause_pred = 1 if state.probs_cause[1] >= state.probs_cause[0] else 0
            total_cause += 1 if cause_pred == cause_label else 0
            total_energy += (state.pred_energy - energy_target) ** 2
            total_spike += state.energy_norm
            steps += 1
        return total_nll, total_cause, total_energy, total_spike, steps

    def evaluate_validation(self, seed: int) -> float:
        if not self.val_sequences:
            return 0.0
        rng = random.Random(seed)
        sequences = list(self.val_sequences)
        rng.shuffle(sequences)
        total_nll = 0.0
        steps = 0
        for record, sequence in sequences:
            unique_chars: set[int] = set()
            action_idx = DOMAIN_TO_ACTION.get(record.domain, 0)
            urgency_val = URGENCY_TO_VALUE.get(record.urgency, 0.55)
            tone_val = TONE_TO_VALUE.get(record.tone, 0.65)
            seq_len = max(len(sequence) - 1, 1)
            for pos in range(len(sequence) - 1):
                current_id = sequence[pos]
                next_id = sequence[pos + 1]
                unique_chars.add(current_id)
                position_ratio = pos / float(seq_len)
                unique_ratio = len(unique_chars) / float(max(len(record.text), 1))
                extras = self._extra_features(position_ratio, unique_ratio, urgency_val, tone_val)
                features = self._assemble_features(current_id, action_idx, extras)
                state = self.self_model.forward(features)
                prob_next = max(state.probs_next[next_id], 1e-8)
                total_nll += -math.log(prob_next)
                steps += 1
        avg_nll = total_nll / float(max(steps, 1))
        return 1.0 / (1.0 + avg_nll)

    def generate_autonomous_samples(self, count: int = 3, max_length: int = 40) -> List[str]:
        samples: List[str] = []
        for _ in range(count):
            domain = self.rng.choice(list(DOMAIN_TO_ACTION.keys()))
            tone = self.rng.choice(list(TONE_TO_VALUE.keys()))
            urgency = self.rng.choice(list(URGENCY_TO_VALUE.keys()))
            samples.append(self._generate_sentence(domain, tone, urgency, max_length))
        return samples

    def _generate_sentence(self, domain: str, tone: str, urgency: str, max_length: int) -> str:
        action_idx = DOMAIN_TO_ACTION.get(domain, 0)
        urgency_val = URGENCY_TO_VALUE.get(urgency, 0.55)
        tone_val = TONE_TO_VALUE.get(tone, 0.65)
        tokens: List[str] = []
        unique_ids: set[int] = set()
        current_id = self.rng.randrange(self.vocab.size)
        seq_len = max_length
        for pos in range(max_length):
            position_ratio = pos / float(max(seq_len - 1, 1))
            unique_ratio = len(unique_ids) / float(max(seq_len, 1))
            extras = self._extra_features(position_ratio, unique_ratio, urgency_val, tone_val)
            features = self._assemble_features(current_id, action_idx, extras)
            state = self.self_model.forward(features)
            next_id = self._sample_from_probs(state.probs_next)
            if next_id == self.vocab.eos_id:
                break
            if next_id == self.vocab.pad_id:
                continue
            token = self.vocab.itos[next_id]
            if token == "<unk>":
                continue
            tokens.append(token)
            unique_ids.add(next_id)
            current_id = next_id
        sentence = "".join(tokens)
        if not sentence:
            sentence = self.vocab.decode([current_id])
        return f"{domain}/{tone}/{urgency}:{sentence}"

    def _assemble_features(self, token_id: int, action_idx: int, extras: Sequence[float]) -> List[float]:
        obs = [0.0 for _ in range(self.vocab.size)]
        obs[token_id] = 1.0
        action_vec = [0.0 for _ in range(self.action_dim)]
        action_vec[action_idx] = 1.0
        return obs + action_vec + list(extras)

    @staticmethod
    def _extra_features(
        position_ratio: float,
        unique_ratio: float,
        urgency_val: float,
        tone_val: float,
    ) -> Tuple[float, float, float, float]:
        return (
            max(0.0, min(position_ratio, 1.0)),
            max(0.0, min(urgency_val, 1.0)),
            max(0.0, min(unique_ratio, 1.0)),
            max(0.0, min(tone_val, 1.0)),
        )

    @staticmethod
    def _targets(
        position_ratio: float,
        tone_val: float,
        urgency_val: float,
        unique_ratio: float,
    ) -> Tuple[float, float]:
        reward = tone_val * 0.8 + 0.15 * (1.0 - position_ratio) + 0.05 * urgency_val
        reward = max(0.0, min(reward, 1.2))
        energy = min(1.0, 0.5 * urgency_val + 0.5 * unique_ratio)
        return reward, energy

    @staticmethod
    def _sample_from_probs(probs: Sequence[float]) -> int:
        threshold = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                return idx
        return len(probs) - 1

    def apply_modification(self, action: str) -> Tuple[bool, str | None]:
        info: str | None = None
        if action == "eta_up":
            self.self_model.hidden_lr = min(self.self_model.hidden_lr * 1.2, 0.2)
            self.self_model.readout_lr = min(self.self_model.readout_lr * 1.2, 0.35)
            info = f"lr={self.self_model.hidden_lr:.3f}/{self.self_model.readout_lr:.3f}"
            return True, info
        if action == "eta_down":
            self.self_model.hidden_lr = max(self.self_model.hidden_lr * 0.8, 0.02)
            self.self_model.readout_lr = max(self.self_model.readout_lr * 0.8, 0.05)
            info = f"lr={self.self_model.hidden_lr:.3f}/{self.self_model.readout_lr:.3f}"
            return True, info
        if action == "vth_up":
            self.self_model.hidden.params.v_th = min(self.self_model.hidden.params.v_th + 0.05, 1.2)
            info = f"v_th={self.self_model.hidden.params.v_th:.2f}"
            return True, info
        if action == "vth_down":
            self.self_model.hidden.params.v_th = max(self.self_model.hidden.params.v_th - 0.05, 0.2)
            info = f"v_th={self.self_model.hidden.params.v_th:.2f}"
            return True, info
        if action == "inner_up":
            if self.self_model.inner_steps >= 24:
                return False, None
            self.self_model.inner_steps += 2
            info = f"inner_steps={self.self_model.inner_steps}"
            return True, info
        if action == "inner_down":
            if self.self_model.inner_steps <= 8:
                return False, None
            self.self_model.inner_steps -= 2
            info = f"inner_steps={self.self_model.inner_steps}"
            return True, info
        if action == "switch_surrogate":
            if self.surrogate_name == "fast_sigmoid":
                self.self_model.hidden.set_surrogate(triangular_surrogate)
                self.surrogate_name = "triangular"
            else:
                self.self_model.hidden.set_surrogate(fast_sigmoid_surrogate)
                self.surrogate_name = "fast_sigmoid"
            return True, self.surrogate_name
        if action == "patch_surrogate":
            def clipped_surrogate(u: float, width: float = 1.1) -> float:
                if abs(u) > width:
                    return 0.0
                return (width - abs(u)) / (width * width)

            self.self_model.hidden.set_surrogate(clipped_surrogate)
            self.surrogate_name = "patched"
            return True, "patched"
        return False, None


def train_minicorpus(config: MiniCorpusConfig | None = None) -> MiniCorpusResult:
    cfg = config or MiniCorpusConfig()
    trainer = MiniCorpusTrainer(cfg)
    meta = MetaLearner(
        window=cfg.meta_window,
        min_delta=cfg.meta_min_delta,
        ab_episodes=3,
        actions=SUPPORTED_ACTIONS,
    )
    history: List[EpisodeMetrics] = []
    score_history: List[float] = []
    last_meta_logs: List[str] = []
    for episode in range(1, cfg.episodes + 1):
        metrics = trainer.train_epoch()
        history.append(metrics)
        score = 1.0 / (1.0 + metrics.avg_nll)
        score_history.append(score)
        logger.info(
            "语料回合 %03d NLL %.3f cause_acc %.2f energy_mse %.3f spikes %.3f",
            episode,
            metrics.avg_nll,
            metrics.cause_acc,
            metrics.energy_mse,
            metrics.mean_spike_norm,
        )
        if meta.should_trigger(score_history):

            def evaluate(candidate: MiniCorpusTrainer, seed: int) -> float:
                return candidate.evaluate_validation(seed)

            trainer, last_meta_logs = meta.adapt(
                trainer,
                step=episode,
                evaluate_fn=evaluate,
            )
            for log_message in last_meta_logs:
                logger.info(log_message)
            score_history.clear()

    tail_window = min(5, len(history))
    if tail_window == 0:
        raise RuntimeError("没有可用的训练历史。")
    tail = history[-tail_window:]
    tail_nll = sum(m.avg_nll for m in tail) / float(tail_window)
    tail_cause = sum(m.cause_acc for m in tail) / float(tail_window)
    tail_energy = sum(m.energy_mse for m in tail) / float(tail_window)
    tail_spike = sum(m.mean_spike_norm for m in tail) / float(tail_window)
    samples = trainer.generate_autonomous_samples()
    for idx, sample in enumerate(samples, 1):
        logger.info("[sample %d] %s", idx, sample)
    return MiniCorpusResult(
        tail_nll=tail_nll,
        tail_cause_acc=tail_cause,
        tail_energy_mse=tail_energy,
        tail_spike_norm=tail_spike,
        generated_samples=samples,
        positive_ratio=meta.positive_ratio(),
        history=history,
    )


__all__ = [
    "MiniCorpusConfig",
    "MiniCorpusResult",
    "train_minicorpus",
]
