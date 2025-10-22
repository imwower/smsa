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
    ) -> float:
        grad = [prob for prob in state.probs]
        grad[target_id] -= 1.0
        grad = [clip(g * advantage, self.clip) for g in grad]

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


def build_corpus(paths: Sequence[str]) -> List[List[str]]:
    lines: List[List[str]] = []
    for path in paths:
        for line in open_stream(path):
            tokens = tokenize(line)
            if tokens:
                lines.append(tokens)
    return lines


def train(args: argparse.Namespace) -> None:
    lines = build_corpus(args.inputs)
    if not lines:
        print("No usable lines found.", file=sys.stderr)
        return

    vocab = Vocab.build(lines, max_size=args.vocab)
    vocab_size = len(vocab.id_to_token)

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

    random.seed(args.seed)

    total_steps = 0
    total_tokens = 0
    sum_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(lines)
        for line_idx, tokens in enumerate(lines, start=1):
            sequence = [TOKEN_BOS] + tokens + [TOKEN_EOS]
            for i in range(len(sequence) - 1):
                current = sequence[i]
                nxt = sequence[i + 1]
                state = model.forward(current)
                loss = model.update(state, vocab.encode(nxt))
                sum_loss += loss
                total_tokens += 1
                total_steps += 1

                if total_tokens % args.report_every == 0:
                    avg_loss = sum_loss / float(total_tokens)
                    ppl = math.exp(min(20.0, avg_loss))
                    print(
                        f"[epoch {epoch}] tokens={total_tokens} loss/token={avg_loss:.3f} ppl={ppl:.2f}"
                    )
                    if args.reset_metrics:
                        sum_loss = 0.0
                        total_tokens = 0
            if args.max_lines and line_idx >= args.max_lines:
                break

    if total_tokens > 0:
        avg_loss = sum_loss / float(total_tokens)
        ppl = math.exp(min(20.0, avg_loss))
        print(f"[final] tokens={total_tokens} loss/token={avg_loss:.3f} ppl={ppl:.2f}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNN-based streaming text LM")
    parser.add_argument("inputs", nargs="+", help="Text files or .gz archives")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--vocab", type=int, default=1024, help="Maximum vocabulary size.")
    parser.add_argument("--input-dim", type=int, default=256, help="Hash projection dimension.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--k-proj", type=int, default=3, help="Hash projections per token.")
    parser.add_argument("--inner-steps", type=int, default=12, help="Inner LIF steps.")
    parser.add_argument("--hidden-lr", type=float, default=0.08, help="Hidden layer learning rate.")
    parser.add_argument("--readout-lr", type=float, default=0.12, help="Readout learning rate.")
    parser.add_argument("--report-every", type=int, default=2000, help="Tokens between reports.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--high-rate", type=float, default=0.95, help="High Poisson rate.")
    parser.add_argument("--low-rate", type=float, default=0.05, help="Low Poisson rate.")
    parser.add_argument("--max-lines", type=int, default=0, help="Optional cap on lines per epoch.")
    parser.add_argument("--reset-metrics", action="store_true", help="Reset metrics after each log.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
