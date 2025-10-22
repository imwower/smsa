"""Streaming text language model using DenseLIF + Linear temporal projection."""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import math
import os
import random
import sys
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

# Ensure repository root on path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snn.dense import DenseLIF, LinearTemporalUnit
from snn.lif import LIFParams, fast_sigmoid_surrogate

TOKEN_BOS = "<bos>"
TOKEN_EOS = "<eos>"
TOKEN_UNK = "<unk>"


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
    return line.strip().split()


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
    tokens: List[List[str]]
    pointer: int = 0
    counts: int = 0
    total_delta: float = 0.0
    history: collections.deque = None  # type: ignore[assignment]
    last_ppl: float = float("inf")

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = collections.deque(maxlen=20)

    def record(self, ppl: float) -> float:
        delta = 0.0
        if self.counts > 0 and math.isfinite(self.last_ppl):
            delta = self.last_ppl - ppl
            self.total_delta += delta
            self.history.append(delta)
        self.counts += 1
        self.last_ppl = ppl
        return delta

    def mean_delta(self) -> float:
        if self.counts == 0:
            return 0.0
        return self.total_delta / float(self.counts)


class FileScheduler:
    def __init__(
        self,
        files_tokens: List[List[List[str]]],
        batch_lines: int,
        ucb_c: float = 0.4,
    ) -> None:
        self.states: List[FileState] = [
            FileState(tokens=toks) for toks in files_tokens if toks
        ]
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
            mean = state.mean_delta()
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
) -> Tuple[float, float, List[float], int]:
    total_loss = 0.0
    total_tokens = 0
    batch_ppls: List[float] = []
    for sequences in batches:
        loss, tokens = run_batch_sequences(model, vocab, list(sequences), train=train)
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
    baseline_model = copy.deepcopy(model)
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


def build_corpora(paths: Sequence[str]) -> List[List[List[str]]]:
    corpora: List[List[List[str]]] = []
    for path in paths:
        tokens_list: List[List[str]] = []
        for line in open_stream(path):
            tokens = tokenize(line)
            if tokens:
                tokens_list.append(tokens)
        if tokens_list:
            corpora.append(tokens_list)
    return corpora


def train(args: argparse.Namespace) -> None:
    corpora = build_corpora(args.inputs)
    if not corpora:
        print("No usable lines found.", file=sys.stderr)
        return

    flattened = [tokens for corpus in corpora for tokens in corpus]
    vocab = Vocab.build(flattened, max_size=args.vocab)
    vocab_size = len(vocab.id_to_token)

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

    random.seed(args.seed)

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
            model, vocab, batches, train=True
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
                f"[train] batches={total_batches} tokens={total_tokens} ppl={report_ppl:.2f}"
            )
            tokens_since_report = 0
            loss_since_report = 0.0

        if detector.update(delta):
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
                if tokens_since_report >= args.report_every:
                    report_loss = loss_since_report / float(max(1, tokens_since_report))
                    report_ppl = math.exp(min(20.0, report_loss))
                    print(
                        f"[train] batches={total_batches} tokens={total_tokens} ppl={report_ppl:.2f}"
                    )
                    tokens_since_report = 0
                    loss_since_report = 0.0
                continue

    overall_loss = cumulative_loss / float(max(1, cumulative_tokens))
    overall_ppl = math.exp(min(20.0, overall_loss)) if cumulative_tokens else 0.0
    print(
        f"[final] batches={total_batches} tokens={total_tokens} ppl={overall_ppl:.2f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNN-based streaming text LM")
    parser.add_argument("inputs", nargs="+", help="Text files or .gz archives")
    parser.add_argument("--vocab", type=int, default=1024, help="Maximum vocabulary size.")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
