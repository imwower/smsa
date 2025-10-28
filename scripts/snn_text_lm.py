"""Streaming text language model using DenseLIF + Linear temporal projection."""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import glob
import gzip
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# Ensure repository root on path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate
from snn.selfmodel import SelfModel

if TYPE_CHECKING:
    from tools.corpus import DomainSampler

TOKEN_BOS = "<bos>"
TOKEN_EOS = "<eos>"
TOKEN_UNK = "<unk>"
SYNTHETIC_LINES = [
    "苹果是一种水果，经常出现在早餐。",
    "蜂蜜属于天然食材，味道清甜。",
    "云计算平台是现代企业的重要工具。",
    "光合作用帮助植物把阳光变成能量。",
    "地球自转让昼夜交替出现。",
    "长江被视为中国的母亲河。",
    "项目计划让团队保持节奏一致。",
    "芝士在烘焙菜单里常被使用。",
]

LM_CSV_PATH = Path("runs/lm.csv")
LM_FIELDS = [
    "timestamp",
    "lines",
    "train_loss",
    "train_ppl",
    "valid_ppl",
    "delta_ppl",
    "spikes",
    "files",
    "topics",
]


def _ensure_lm_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LM_FIELDS)
            writer.writeheader()


def _read_last_valid_ppl() -> Optional[float]:
    if not LM_CSV_PATH.exists():
        return None
    try:
        with LM_CSV_PATH.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            last_row = None
            for row in reader:
                last_row = row
    except (OSError, csv.Error):
        return None
    if not last_row:
        return None
    value = last_row.get("valid_ppl")
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _format_counts(counts: Mapping[str, int], *, short: bool = False) -> str:
    if not counts:
        return "synthetic"
    items = []
    for key, value in sorted(counts.items(), key=lambda item: item[0]):
        label = Path(key).name if short else key
        items.append(f"{label}:{value}")
    return "|".join(items)


def _append_lm_row(
    *,
    lines: int,
    avg_loss: float,
    train_ppl: float,
    valid_ppl: float,
    delta_ppl: float,
    avg_spikes: float,
    files: Mapping[str, int],
    topics: Mapping[str, int],
) -> None:
    _ensure_lm_csv(LM_CSV_PATH)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row = {
        "timestamp": timestamp,
        "lines": lines,
        "train_loss": f"{avg_loss:.6f}",
        "train_ppl": f"{train_ppl:.6f}",
        "valid_ppl": f"{valid_ppl:.6f}",
        "delta_ppl": f"{delta_ppl:.6f}",
        "spikes": f"{avg_spikes:.6f}",
        "files": _format_counts(files, short=True),
        "topics": _format_counts(topics, short=True),
    }
    with LM_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LM_FIELDS)
        writer.writerow(row)


def _load_validation_sequences(state_path: Path, limit: int = 128) -> List[List[str]]:
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    files = [
        Path(path)
        for path, meta in state.get("files", {}).items()
        if meta.get("split") == "valid"
    ]
    if not files:
        return []
    per_file = max(1, limit // len(files))
    sequences: List[List[str]] = []
    for file_path in files:
        try:
            opener = gzip.open if file_path.suffix == ".gz" else open
            with opener(file_path, "rt", encoding="utf-8", errors="ignore") as handle:
                for idx, line in enumerate(handle):
                    tokens = tokenize(line.rstrip("\n"))
                    if tokens:
                        sequences.append(tokens)
                    if idx + 1 >= per_file or len(sequences) >= limit:
                        break
        except OSError:
            continue
        if len(sequences) >= limit:
            break
    return sequences


def _synthetic_valid_sequences(count: int = 32) -> List[List[str]]:
    seqs: List[List[str]] = []
    for idx in range(count):
        text = SYNTHETIC_LINES[idx % len(SYNTHETIC_LINES)]
        seqs.append(tokenize(text))
    return seqs


def _evaluate_ppl(
    model: TextSNNLM,
    vocab: Vocab,
    sequences: Sequence[Sequence[str]],
) -> float:
    if not sequences:
        return float("inf")
    total_loss = 0.0
    total_tokens = 0
    for tokens in sequences:
        seq = [TOKEN_BOS] + list(tokens) + [TOKEN_EOS]
        model.reset_temporal()
        for idx in range(len(seq) - 1):
            state = model.forward(seq[idx])
            loss = model.update(state, vocab.encode(seq[idx + 1]), train=False)
            total_loss += loss
            total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    avg = total_loss / float(total_tokens)
    return math.exp(min(20.0, avg))


def expand_inputs(patterns: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(pattern)
    return resolved


def open_stream(path: str) -> Iterator[str]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                yield line.rstrip("\n")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                yield line.rstrip("\n")


def tokenize(line: str) -> List[str]:
    if not line:
        return []
    return list(line)


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    unk_id: int

    @classmethod
    def build(cls, corpus: Iterable[List[str]], max_size: int) -> Vocab:
        counter: Dict[str, int] = collections.Counter()
        for tokens in corpus:
            counter.update(tokens)
        specials = [TOKEN_BOS, TOKEN_EOS, TOKEN_UNK]
        most_common = [
            tok for tok, _ in counter.most_common(max(0, max_size - len(specials)))
            if tok not in specials
        ]
        id_to_token = specials + most_common
        token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, unk_id=token_to_id[TOKEN_UNK])

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)


class HashProjector:
    def __init__(self, n_in: int, k: int = 3) -> None:
        self.n_in = n_in
        self.k = k

    def indices(self, token: str) -> List[int]:
        indices: List[int] = []
        used = set()
        for probe in range(self.k * 2):
            seed = sha256(f"{token}:{probe}".encode("utf-8")).hexdigest()
            val = int(seed, 16) % self.n_in
            if val in used:
                continue
            used.add(val)
            indices.append(val)
            if len(indices) >= self.k:
                break
        while len(indices) < self.k:
            indices.append(random.randrange(self.n_in))
        return indices

    def rates(self, token: str, high: float, low: float) -> List[float]:
        base = [low] * self.n_in
        for idx in self.indices(token):
            base[idx] = high
        return base


@dataclass
class FileState:
    label: str
    tokens: List[List[str]]
    pointer: int = 0
    counts: int = 0
    total_delta: float = 0.0
    history: Optional[Deque[float]] = None
    last_ppl: float = float("inf")

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = collections.deque(maxlen=20)

    def record(self, ppl: float) -> float:
        delta = 0.0
        if self.counts > 0 and math.isfinite(self.last_ppl):
            delta = self.last_ppl - ppl
            self.total_delta += delta
            if self.history is not None:
                self.history.append(delta)
        self.counts += 1
        self.last_ppl = ppl
        return delta

    def mean_delta(self) -> float:
        if self.counts == 0:
            return 0.0
        return self.total_delta / float(self.counts)

    def recent_delta(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / float(len(self.history))


class FileScheduler:
    def __init__(
        self,
        corpora: Sequence[Tuple[str, List[List[str]]]],
        batch_lines: int,
        ucb_c: float = 0.4,
    ) -> None:
        self.states: List[FileState] = []
        for label, tokens in corpora:
            if tokens:
                self.states.append(FileState(label=label, tokens=tokens))
        if not self.states:
            raise ValueError("No non-empty files available.")
        self.batch_lines = batch_lines
        self.total_batches = 0
        self.ucb_c = ucb_c

    def select_file(self) -> int:
        for idx, state in enumerate(self.states):
            if state.counts == 0:
                return idx
        total = max(1, self.total_batches)
        best_idx = 0
        best_score = -float("inf")
        for idx, state in enumerate(self.states):
            mean = state.recent_delta()
            bonus = self.ucb_c * math.sqrt(
                2.0 * math.log(total + 1) / max(1, state.counts)
            )
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _collect_batch(
        self, state: FileState, pointer: int
    ) -> Tuple[List[List[str]], int]:
        batch: List[List[str]] = []
        tokens = state.tokens
        if not tokens:
            return batch, pointer
        for _ in range(self.batch_lines):
            batch.append(tokens[pointer])
            pointer = (pointer + 1) % len(tokens)
        return batch, pointer

    def fetch_batches(
        self, file_idx: int, num_batches: int, advance: bool
    ) -> Tuple[List[List[List[str]]], int]:
        state = self.states[file_idx]
        pointer = state.pointer
        batches: List[List[List[str]]] = []
        for _ in range(num_batches):
            batch, pointer = self._collect_batch(state, pointer)
            batches.append(batch)
        if advance:
            state.pointer = pointer
            self.total_batches += num_batches
        return batches, pointer

    def update_stats(self, file_idx: int, ppl_values: Sequence[float]) -> float:
        state = self.states[file_idx]
        delta = 0.0
        for ppl in ppl_values:
            delta = state.record(ppl)
        return delta

    def set_pointer(self, file_idx: int, pointer: int) -> None:
        self.states[file_idx].pointer = pointer

    def label_for(self, file_idx: int) -> str:
        return self.states[file_idx].label

    def summaries(self) -> List[str]:
        summary: List[str] = []
        for state in self.states:
            if not state.history:
                continue
            summary.append(f"{state.label}:{state.recent_delta():+.3f}")
        return summary

    def recent_delta(self, file_idx: int) -> float:
        return self.states[file_idx].recent_delta()


class PlateauDetector:
    def __init__(self, window: int, min_delta: float, cooldown: int) -> None:
        self.window = window
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.delta_history: collections.deque = collections.deque(maxlen=window)
        self.cooldown_timer = 0

    def update(self, delta: float) -> bool:
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        self.delta_history.append(delta)
        if (
            len(self.delta_history) == self.window
            and self.cooldown_timer == 0
            and sum(self.delta_history) < self.min_delta
        ):
            self.delta_history.clear()
            self.cooldown_timer = self.cooldown
            return True
        return False


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self._data: Deque[Tuple[str, ...]] = collections.deque(maxlen=capacity)

    def add(self, sequence: Sequence[str]) -> None:
        if not sequence:
            return
        self._data.append(tuple(sequence))

    def sample(self, count: int) -> List[List[str]]:
        if count <= 0 or not self._data:
            return []
        count = min(count, len(self._data))
        data_list = list(self._data)
        indices = self.rng.sample(range(len(data_list)), count)
        return [list(data_list[idx]) for idx in indices]

    def random_choice(self) -> Optional[List[str]]:
        if not self._data:
            return None
        data_list = list(self._data)
        return list(self.rng.choice(data_list))

    def __len__(self) -> int:
        return len(self._data)


class DreamSelfModel:
    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        *,
        hidden_size: int = 64,
        inner_steps: int = 12,
        seed: int = 0,
    ) -> None:
        params = LIFParams(v_th=0.5, tau_m=8.5, tau_a=17.0, beta=0.35, refractory=2)
        self.model = SelfModel(
            obs_dim=vocab_size,
            action_dim=0,
            hidden_size=hidden_size,
            lif_params=params,
            inner_steps=inner_steps,
            hidden_lr=0.08,
            readout_lr=0.15,
            clip=1.5,
            obs_weight=1.0,
            reward_weight=0.01,
            energy_weight=0.01,
            cause_weight=0.01,
        )
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.extra_features = [0.0, 0.0, 0.0, 0.0]
        self.rng = random.Random(seed)

    def _features(self, token_id: int) -> List[float]:
        features = [0.0 for _ in range(self.vocab_size)]
        if 0 <= token_id < self.vocab_size:
            features[token_id] = 1.0
        return features + self.extra_features

    def update_pair(self, prev_id: int, next_id: int) -> None:
        features = self._features(prev_id)
        state = self.model.forward(features)
        self.model.update(
            state=state,
            next_obs_index=max(0, min(next_id, self.vocab_size - 1)),
            reward_target=0.0,
            energy_target=0.0,
            cause_label=0,
        )

    def dream_sequence(self, start_id: int, max_steps: int) -> List[int]:
        prev = start_id
        sequence: List[int] = []
        for _ in range(max_steps):
            features = self._features(prev)
            state = self.model.forward(features)
            next_id = self._sample(state.probs_next)
            if next_id == self.eos_id:
                break
            if next_id == self.bos_id:
                prev = next_id
                continue
            sequence.append(next_id)
            prev = next_id
        return sequence

    def _sample(self, probs: Sequence[float]) -> int:
        if not probs:
            return self.rng.randrange(self.vocab_size)
        total = sum(probs)
        if total <= 0.0 or not math.isfinite(total):
            return self.rng.randrange(self.vocab_size)
        target = self.rng.random()
        accum = 0.0
        for idx, prob in enumerate(probs):
            accum += prob / total
            if target <= accum:
                return idx
        return len(probs) - 1


class DreamHelper:
    def __init__(
        self,
        vocab: Vocab,
        *,
        capacity: int,
        replay_ratio: float,
        dream_ratio: float,
        min_fill: int,
        min_len: int,
        max_len: int,
        seed: int,
    ) -> None:
        self.vocab = vocab
        self.replay_ratio = max(0.0, replay_ratio)
        self.dream_ratio = max(0.0, dream_ratio)
        self.min_fill = max(1, min_fill)
        self.min_len = max(1, min_len)
        self.max_len = max(self.min_len, max_len)
        self.bos_id = vocab.encode(TOKEN_BOS)
        self.eos_id = vocab.encode(TOKEN_EOS)
        self.buffer_rng = random.Random(seed)
        self.buffer = ReplayBuffer(capacity, self.buffer_rng)
        self.model = DreamSelfModel(
            vocab_size=len(vocab.id_to_token),
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            seed=seed + 1,
        )

    def augment(self, sequences: List[List[str]]) -> List[List[str]]:
        if not sequences:
            return []
        augmented = [list(seq) for seq in sequences]
        for seq in sequences:
            self._update_model_for_sequence(seq)
            if seq:
                self.buffer.add(seq)
        if len(self.buffer) < self.min_fill:
            return augmented

        extras: List[List[str]] = []
        if self.replay_ratio > 0.0:
            num_replay = max(1, int(round(len(sequences) * self.replay_ratio)))
            extras.extend(self.buffer.sample(num_replay))
        if self.dream_ratio > 0.0:
            num_dream = max(1, int(round(len(sequences) * self.dream_ratio)))
            extras.extend(self._dream_samples(num_dream))

        augmented.extend([seq for seq in extras if seq])
        return augmented

    def _update_model_for_sequence(self, tokens: Sequence[str]) -> None:
        prev_id = self.bos_id
        if tokens:
            for token in tokens:
                token_id = self.vocab.encode(token)
                self.model.update_pair(prev_id, token_id)
                prev_id = token_id
        self.model.update_pair(prev_id, self.eos_id)

    def _dream_samples(self, count: int) -> List[List[str]]:
        samples: List[List[str]] = []
        for _ in range(count):
            seed_seq = self.buffer.random_choice()
            if seed_seq is None:
                break
            start_token = TOKEN_BOS
            if seed_seq:
                choices = [TOKEN_BOS] + seed_seq
                start_token = self.buffer_rng.choice(choices)
            start_id = self.vocab.encode(start_token)
            length = self.buffer_rng.randint(self.min_len, self.max_len)
            token_ids = self.model.dream_sequence(start_id, length)
            if not token_ids:
                continue
            tokens = [
                self.vocab.id_to_token[token_id]
                for token_id in token_ids
                if token_id not in (self.bos_id, self.eos_id)
            ]
            if tokens:
                samples.append(tokens)
        return samples


def apply_action(model: TextSNNLM, action: str) -> Tuple[bool, str | None]:
    info: str | None = None
    if action == "eta_up":
        model.hidden_lr = min(model.hidden_lr * 1.2, 0.5)
        info = f"hidden_lr={model.hidden_lr:.4f}"
        return True, info
    if action == "eta_down":
        model.hidden_lr = max(model.hidden_lr * 0.8, 0.01)
        info = f"hidden_lr={model.hidden_lr:.4f}"
        return True, info
    if action == "vth_up":
        model.hidden.params.v_th = min(model.hidden.params.v_th + 0.05, 1.2)
        info = f"v_th={model.hidden.params.v_th:.3f}"
        return True, info
    if action == "vth_down":
        model.hidden.params.v_th = max(model.hidden.params.v_th - 0.05, 0.2)
        info = f"v_th={model.hidden.params.v_th:.3f}"
        return True, info
    if action == "inner_up":
        model.inner_steps = min(model.inner_steps + 4, 40)
        info = f"inner_steps={model.inner_steps}"
        return True, info
    if action == "inner_down":
        if model.inner_steps <= 6:
            return False, info
        model.inner_steps = max(model.inner_steps - 4, 4)
        info = f"inner_steps={model.inner_steps}"
        return True, info
    return False, info


def run_batch_sequences(
    model: TextSNNLM,
    vocab: Vocab,
    sequences: List[List[str]],
    train: bool,
) -> Tuple[float, int]:
    total_loss = 0.0
    total_tokens = 0
    for tokens in sequences:
        seq_tokens = [TOKEN_BOS] + tokens + [TOKEN_EOS]
        model.reset_temporal()
        for i in range(len(seq_tokens) - 1):
            current = seq_tokens[i]
            nxt = seq_tokens[i + 1]
            state = model.forward(current)
            loss = model.update(state, vocab.encode(nxt), train=train)
            total_loss += loss
            total_tokens += 1
    return total_loss, total_tokens


def run_batches(
    model: TextSNNLM,
    vocab: Vocab,
    batches: Sequence[Sequence[List[str]]],
    train: bool,
    dream_helper: Optional[DreamHelper] = None,
) -> Tuple[float, float, List[float], int]:
    total_loss = 0.0
    total_tokens = 0
    batch_ppls: List[float] = []
    for sequences in batches:
        seq_list = [list(seq) for seq in sequences]
        if dream_helper is not None and train:
            seq_list = dream_helper.augment(seq_list)
        loss, tokens = run_batch_sequences(model, vocab, seq_list, train=train)
        total_loss += loss
        total_tokens += tokens
        avg_loss = loss / float(max(1, tokens))
        batch_ppls.append(math.exp(min(20.0, avg_loss)))
    avg_loss = total_loss / float(max(1, total_tokens))
    avg_ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, avg_ppl, batch_ppls, total_tokens


def adapt_parameters(
    model: TextSNNLM,
    vocab: Vocab,
    scheduler: FileScheduler,
    file_idx: int,
    batches: Sequence[Sequence[List[str]]],
    pointer_after: int,
    step: int,
) -> Tuple[TextSNNLM, bool, List[str], int, float]:
    messages: List[str] = []
    _, base_ppl, _, _ = run_batches(copy.deepcopy(model), vocab, batches, train=False)
    actions = [
        "eta_up",
        "eta_down",
        "vth_up",
        "vth_down",
        "inner_up",
        "inner_down",
    ]
    for action in actions:
        candidate = copy.deepcopy(model)
        applied, info = apply_action(candidate, action)
        if not applied:
            messages.append(
                f"[meta] step {step:05d} action={action} skipped (invalid)"
            )
            continue
        cand_loss, cand_ppl, batch_ppls, cand_tokens = run_batches(
            candidate, vocab, batches, train=True
        )
        delta = base_ppl - cand_ppl
        if delta > 0.0:
            scheduler.set_pointer(file_idx, pointer_after)
            scheduler.total_batches += len(batches)
            scheduler.update_stats(file_idx, batch_ppls)
            messages.append(
                f"[meta] step {step:05d} action={action} delta={delta:.4f} (accepted, info={info})"
            )
            return candidate, True, messages, cand_tokens, cand_loss
        messages.append(
            f"[meta] step {step:05d} action={action} delta={delta:.4f} (reverted, info={info})"
        )
    return model, False, messages, 0, 0.0


def softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    apex = max(logits)
    exp_vals = [math.exp(val - apex) for val in logits]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [val / denom for val in exp_vals]


def spike_from_rate(rate: float) -> int:
    rate = max(0.0, min(rate, 1.0))
    return 1 if random.random() < rate else 0


def clip(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass
class LMState:
    probs: List[float]
    hidden_rates: List[float]
    eligibility_history: List[List[List[float]]]
    bias_history: List[List[float]]


class TextSNNLM:
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 256,
        hidden_size: int = 128,
        k_proj: int = 3,
        inner_steps: int = 12,
        hidden_lr: float = 0.08,
        readout_lr: float = 0.12,
        clip_value: float = 2.0,
        high_rate: float = 0.95,
        low_rate: float = 0.05,
    ) -> None:
        params = LIFParams(v_th=0.52, tau_m=9.0, tau_a=18.0, beta=0.4, refractory=2)
        self.hidden = DenseLIF(
            n_in=input_dim * 2,
            n_out=hidden_size,
            params=params,
            surrogate_fn=fast_sigmoid_surrogate,
        )
        self.temporal = LinearTemporalUnit(n_in=input_dim, n_state=input_dim, beta=0.9)
        self.projector = HashProjector(n_in=input_dim, k=k_proj)
        self.inner_steps = inner_steps
        self.hidden_lr = hidden_lr
        self.readout_lr = readout_lr
        self.clip = clip_value
        self.high_rate = high_rate
        self.low_rate = low_rate
        self.readout_weights = [
            [random.uniform(-0.15, 0.15) for _ in range(vocab_size)]
            for _ in range(hidden_size)
        ]
        self.readout_bias = [0.0 for _ in range(vocab_size)]

    def rates_for_token(self, token: str) -> List[float]:
        return self.projector.rates(token, self.high_rate, self.low_rate)

    def reset_temporal(self) -> None:
        self.temporal.reset()

    def forward(self, token: str) -> LMState:
        base_rates = self.rates_for_token(token)
        temporal_state = self.temporal.transform(base_rates)
        combined_rates = base_rates + [max(0.0, min(val, 1.0)) for val in temporal_state]

        self.hidden.reset_state()
        hidden_counts = [0 for _ in range(self.hidden.n_out)]
        eligibility_history: List[List[List[float]]] = []
        bias_history: List[List[float]] = []

        for _ in range(self.inner_steps):
            spikes = [spike_from_rate(rate) for rate in combined_rates]
            spike_out, _, eligibility_snapshot, bias_snapshot = self.hidden.step(spikes)
            hidden_counts = [count + spike for count, spike in zip(hidden_counts, spike_out)]
            eligibility_history.append([row[:] for row in eligibility_snapshot])
            bias_history.append(bias_snapshot[:])

        hidden_rates = [count / float(self.inner_steps) for count in hidden_counts]
        logits = []
        for vocab_idx in range(len(self.readout_bias)):
            logit = self.readout_bias[vocab_idx]
            for h in range(self.hidden.n_out):
                logit += self.readout_weights[h][vocab_idx] * hidden_rates[h]
            logits.append(logit)
        probs = softmax(logits)
        return LMState(
            probs=probs,
            hidden_rates=hidden_rates,
            eligibility_history=eligibility_history,
            bias_history=bias_history,
        )

    def learning_signal(self, grad: Sequence[float]) -> List[float]:
        signals = [0.0 for _ in range(self.hidden.n_out)]
        for idx, g in enumerate(grad):
            for h in range(self.hidden.n_out):
                signals[h] += g * self.readout_weights[h][idx]
        return [clip(val, self.clip) for val in signals]

    def update(
        self,
        state: LMState,
        target_id: int,
        advantage: float = 1.0,
        train: bool = True,
    ) -> float:
        grad = [prob for prob in state.probs]
        grad[target_id] -= 1.0
        grad = [clip(g * advantage, self.clip) for g in grad]

        if train:
            signals = self.learning_signal(grad)
            self.hidden.eprop_apply(signals, self.hidden_lr)

            for h in range(self.hidden.n_out):
                rate = state.hidden_rates[h]
                for vocab_idx, g in enumerate(grad):
                    delta = clip(g * rate, self.clip)
                    self.readout_weights[h][vocab_idx] -= self.readout_lr * delta
            for vocab_idx, g in enumerate(grad):
                self.readout_bias[vocab_idx] -= self.readout_lr * clip(g, self.clip)

        loss = -math.log(max(state.probs[target_id], 1e-8))
        return loss


def build_corpora(paths: Sequence[str]) -> List[Tuple[str, List[List[str]]]]:
    corpora: List[Tuple[str, List[List[str]]]] = []
    for path in paths:
        tokens_list: List[List[str]] = []
        try:
            for line in open_stream(path):
                tokens = tokenize(line)
                if tokens:
                    tokens_list.append(tokens)
        except OSError as exc:
            print(f"[warn] skip {path}: {exc}", file=sys.stderr)
            continue
        if tokens_list:
            label = os.path.basename(path)
            corpora.append((label, tokens_list))
    return corpora


def train(args: argparse.Namespace) -> None:
    patterns = args.inputs if args.inputs else args.corpus_glob
    if not patterns:
        print("No corpus patterns provided.", file=sys.stderr)
        return
    input_paths = expand_inputs(patterns)
    corpora = build_corpora(input_paths)
    if not corpora:
        print("No usable lines found.", file=sys.stderr)
        return

    flattened = [tokens for _, corpus in corpora for tokens in corpus]
    vocab = Vocab.build(flattened, max_size=args.vocab_size)
    vocab_size = len(vocab.id_to_token)

    random.seed(args.seed)

    scheduler = FileScheduler(corpora, batch_lines=args.batch_lines, ucb_c=args.ucb_c)
    model = TextSNNLM(
        vocab_size=vocab_size,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        k_proj=args.k_proj,
        inner_steps=args.inner_steps,
        hidden_lr=args.hidden_lr,
        readout_lr=args.readout_lr,
        high_rate=args.high_rate,
        low_rate=args.low_rate,
    )

    detector = PlateauDetector(
        window=args.plateau_window,
        min_delta=args.plateau_delta,
        cooldown=args.plateau_cooldown,
    )

    dream_helper: Optional[DreamHelper] = None
    # --dream on/off overrides dream usage (off => no dream augmentation)
    if getattr(args, "dream", "off") == "off":
        args.dream_ratio = 0.0
    if args.replay_ratio > 0.0 or args.dream_ratio > 0.0:
        dream_helper = DreamHelper(
            vocab=vocab,
            capacity=max(args.replay_capacity, args.batch_lines),
            replay_ratio=max(0.0, args.replay_ratio),
            dream_ratio=max(0.0, args.dream_ratio),
            min_fill=max(args.replay_warmup, args.batch_lines),
            min_len=args.dream_min_len,
            max_len=args.dream_max_len,
            seed=args.seed + 17,
        )

    total_tokens = 0
    total_batches = 0
    loss_since_report = 0.0
    tokens_since_report = 0
    cumulative_loss = 0.0
    cumulative_tokens = 0

    while True:
        if args.max_batches and total_batches >= args.max_batches:
            break
        if args.max_tokens and total_tokens >= args.max_tokens:
            break

        file_idx = scheduler.select_file()
        batches, pointer_after = scheduler.fetch_batches(file_idx, 1, advance=False)
        if not batches or not batches[0]:
            scheduler.set_pointer(file_idx, pointer_after)
            scheduler.total_batches += 1
            total_batches += 1
            continue

        avg_loss, avg_ppl, batch_ppls, tokens = run_batches(
            model, vocab, batches, train=True, dream_helper=dream_helper
        )
        scheduler.set_pointer(file_idx, pointer_after)
        scheduler.total_batches += 1
        delta = scheduler.update_stats(file_idx, batch_ppls)

        total_tokens += tokens
        tokens_since_report += tokens
        loss_since_report += avg_loss * tokens
        cumulative_loss += avg_loss * tokens
        cumulative_tokens += tokens
        total_batches += 1

        if tokens_since_report >= args.report_every:
            report_loss = loss_since_report / float(max(1, tokens_since_report))
            report_ppl = math.exp(min(20.0, report_loss))
            print(
                f"[train] batches={total_batches} tokens={total_tokens} loss/token={report_loss:.4f} ppl={report_ppl:.2f}"
            )
            domain_summary = scheduler.summaries()
            if domain_summary:
                print("[meta] recent Δppl " + " ".join(domain_summary))
            tokens_since_report = 0
            loss_since_report = 0.0

        if detector.update(delta):
            print(
                f"[meta] plateau file={scheduler.label_for(file_idx)} recentΔ={scheduler.recent_delta(file_idx):+.4f}"
            )
            preview_batches, pointer_preview = scheduler.fetch_batches(
                file_idx, args.adapt_batches, advance=False
            )
            model, accepted, messages, adapt_tokens, adapt_loss = adapt_parameters(
                model,
                vocab,
                scheduler,
                file_idx,
                preview_batches,
                pointer_preview,
                total_batches,
            )
            for msg in messages:
                print(msg)
            if accepted:
                consumed_batches = len(preview_batches)
                total_batches += consumed_batches
                total_tokens += adapt_tokens
                tokens_since_report += adapt_tokens
                total_loss_contrib = adapt_loss * adapt_tokens
                loss_since_report += total_loss_contrib
                cumulative_loss += total_loss_contrib
                cumulative_tokens += adapt_tokens
                if tokens_since_report >= args.report_every:
                    report_loss = loss_since_report / float(max(1, tokens_since_report))
                    report_ppl = math.exp(min(20.0, report_loss))
                    print(
                        f"[train] batches={total_batches} tokens={total_tokens} loss/token={report_loss:.4f} ppl={report_ppl:.2f}"
                    )
                    domain_summary = scheduler.summaries()
                    if domain_summary:
                        print("[meta] recent Δppl " + " ".join(domain_summary))
                    tokens_since_report = 0
                    loss_since_report = 0.0
                continue

    overall_loss = cumulative_loss / float(max(1, cumulative_tokens))
    overall_ppl = math.exp(min(20.0, overall_loss)) if cumulative_tokens else 0.0
    print(
        f"[final] batches={total_batches} tokens={total_tokens} loss/token={overall_loss:.4f} ppl={overall_ppl:.2f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNN-based streaming text LM")
    parser.add_argument(
        "--corpus-glob",
        nargs="*",
        default=["data/*.txt", "data/*.txt.gz", "data/*.gz"],
        help="Glob patterns for streaming text corpora (.txt or .gz).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit file paths (overrides glob).",
    )
    parser.add_argument("--vocab-size", type=int, default=1024, help="Maximum vocabulary size (includes specials).")
    parser.add_argument("--vocab", type=int, dest="vocab_size", help=argparse.SUPPRESS)
    parser.add_argument("--input-dim", type=int, default=256, help="Hash projection dimension.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--k-proj", type=int, default=3, help="Hash projections per token.")
    parser.add_argument("--inner-steps", type=int, default=12, help="Inner LIF steps.")
    parser.add_argument("--hidden-lr", type=float, default=0.08, help="Hidden layer learning rate.")
    parser.add_argument("--readout-lr", type=float, default=0.12, help="Readout learning rate.")
    parser.add_argument("--batch-lines", type=int, default=32, help="Lines per training batch.")
    parser.add_argument("--report-every", type=int, default=2000, help="Tokens between reports.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--high-rate", type=float, default=0.95, help="High Poisson rate.")
    parser.add_argument("--low-rate", type=float, default=0.05, help="Low Poisson rate.")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after this many batches (0 for unlimited).")
    parser.add_argument("--max-tokens", type=int, default=0, help="Stop after this many tokens (0 for unlimited).")
    parser.add_argument("--ucb-c", type=float, default=0.4, help="UCB exploration constant.")
    parser.add_argument("--plateau-window", type=int, default=20, help="Window size for plateau detection.")
    parser.add_argument("--plateau-delta", type=float, default=0.02, help="Minimum cumulative Δppl to avoid plateau.")
    parser.add_argument("--plateau-cooldown", type=int, default=40, help="Cooldown batches after a plateau adaption.")
    parser.add_argument("--adapt-batches", type=int, default=5, help="Number of batches for A/B adaptation.")
    parser.add_argument("--replay-capacity", type=int, default=2048, help="Replay buffer capacity.")
    parser.add_argument("--replay-ratio", type=float, default=0.3, help="Replay sequences per batch ratio.")
    parser.add_argument("--dream-ratio", type=float, default=0.2, help="Dream sequences per batch ratio.")
    parser.add_argument("--dream", type=str, choices=["on", "off"], default="off", help="Toggle dream augmentation on/off.")
    parser.add_argument("--dream-min-len", type=int, default=3, help="Minimum dream sequence length.")
    parser.add_argument("--dream-max-len", type=int, default=5, help="Maximum dream sequence length.")
    parser.add_argument("--replay-warmup", type=int, default=64, help="Minimum sequences before enabling replay/dream.")
    return parser.parse_args(argv)


def train_lines(
    num_lines: int,
    seed: int | None = None,
    *,
    sampler: "DomainSampler | None" = None,
    valid_interval: int = 200,
    log_path: str | Path | None = None,
) -> Dict[str, object]:
    """Lightweight LM training used by daemon loops."""
    global LM_CSV_PATH
    if log_path is not None:
        LM_CSV_PATH = Path(log_path)
    rng = random.Random(seed)
    target_lines = max(1, num_lines)
    collected: List[List[str]] = []
    file_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    file_topics: Dict[str, str] = {}

    while len(collected) < target_lines:
        remaining = target_lines - len(collected)
        chunk = min(64, remaining)
        if sampler is None:
            sample = tokenize(rng.choice(SYNTHETIC_LINES))
            collected.append(sample)
            file_counts["synthetic"] = file_counts.get("synthetic", 0) + 1
            topic_counts["synthetic"] = topic_counts.get("synthetic", 0) + 1
            continue
        try:
            # 兼容新版 DomainSampler.next_lines()
            if hasattr(sampler, "next_lines"):
                batch_lines = list(sampler.next_lines(chunk))  # type: ignore[attr-defined]
            else:
                batch_lines = list(sampler.next_batch(chunk))
        except RuntimeError:
            sampler = None
            continue
        if not batch_lines:
            sampler = None
            continue
        try:
            meta = sampler.metadata_for(sampler.last_file())
        except Exception:
            meta = {}
        file_path = meta.get("path") or sampler.last_file() or "unknown"
        topic = meta.get("topic", "default")
        file_counts[file_path] = file_counts.get(file_path, 0) + len(batch_lines)
        topic_counts[topic] = topic_counts.get(topic, 0) + len(batch_lines)
        file_topics[file_path] = topic
        for line in batch_lines:
            tokens = tokenize(line)
            if tokens:
                collected.append(tokens)
        if len(collected) >= target_lines:
            break

    if not collected:
        collected = [tokenize(rng.choice(SYNTHETIC_LINES)) for _ in range(target_lines)]
        file_counts = {"synthetic": target_lines}
        topic_counts = {"synthetic": target_lines}
        file_topics = {"synthetic": "synthetic"}

    vocab = Vocab.build(collected, max_size=160)
    model = TextSNNLM(
        vocab_size=len(vocab.id_to_token),
        input_dim=96,
        hidden_size=48,
        k_proj=2,
        inner_steps=8,
        hidden_lr=0.05,
        readout_lr=0.08,
    )

    state_attr = getattr(sampler, "state_path", None)
    state_path = Path(state_attr) if state_attr else None
    valid_sequences: List[List[str]] = []
    if state_path and state_path.exists():
        valid_sequences = _load_validation_sequences(state_path)
    if not valid_sequences:
        valid_sequences = _synthetic_valid_sequences(min(32, len(collected)))

    total_loss = 0.0
    total_tokens = 0
    total_spikes = 0.0
    lines_trained = 0
    lines_since_eval = 0
    valid_interval = max(1, min(valid_interval, target_lines))
    prev_logged_ppl = _read_last_valid_ppl()
    last_valid = None
    last_delta = 0.0

    for tokens in collected:
        seq = [TOKEN_BOS] + tokens + [TOKEN_EOS]
        model.reset_temporal()
        for idx in range(len(seq) - 1):
            state = model.forward(seq[idx])
            total_spikes += sum(state.hidden_rates) * model.inner_steps
            loss = model.update(state, vocab.encode(seq[idx + 1]), train=True)
            total_loss += loss
            total_tokens += 1
        lines_trained += 1
        lines_since_eval += 1
        if lines_since_eval >= valid_interval:
            last_valid = _evaluate_ppl(model, vocab, valid_sequences)
            avg_loss = total_loss / float(max(1, total_tokens))
            train_ppl = math.exp(min(20.0, avg_loss))
            avg_spikes_mid = total_spikes / float(max(1, total_tokens))
            delta = (
                (prev_logged_ppl - last_valid)
                if (prev_logged_ppl is not None and math.isfinite(last_valid))
                else 0.0
            )
            _append_lm_row(
                lines=lines_trained,
                avg_loss=avg_loss,
                train_ppl=train_ppl,
                valid_ppl=last_valid,
                delta_ppl=delta,
                avg_spikes=avg_spikes_mid,
                files=file_counts,
                topics=topic_counts,
            )
            prev_logged_ppl = last_valid
            last_delta = delta
            lines_since_eval = 0

    if lines_since_eval > 0 or last_valid is None:
        last_valid = _evaluate_ppl(model, vocab, valid_sequences)
        avg_loss = total_loss / float(max(1, total_tokens))
        train_ppl = math.exp(min(20.0, avg_loss))
        avg_spikes_mid = total_spikes / float(max(1, total_tokens))
        delta = (
            (prev_logged_ppl - last_valid)
            if (prev_logged_ppl is not None and math.isfinite(last_valid))
            else 0.0
        )
        _append_lm_row(
            lines=lines_trained,
            avg_loss=avg_loss,
            train_ppl=train_ppl,
            valid_ppl=last_valid,
            delta_ppl=delta,
            avg_spikes=avg_spikes_mid,
            files=file_counts,
            topics=topic_counts,
        )
        prev_logged_ppl = last_valid
        last_delta = delta

    avg_loss = total_loss / float(max(1, total_tokens))
    train_ppl = math.exp(min(20.0, avg_loss))
    avg_spikes = total_spikes / float(max(1, total_tokens))
    return {
        "avg_loss": avg_loss,
        "ppl": train_ppl,
        "avg_spikes": avg_spikes,
        "valid_ppl": last_valid if last_valid is not None else float("inf"),
        "delta_ppl": last_delta,
        "files": file_counts,
        "topics": topic_counts,
        "file_topics": file_topics,
        "lines": lines_trained,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
