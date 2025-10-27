"""基于尖峰神经网络的文本生成器，复用 TextSNNLM 读出模块进行采样。"""

from __future__ import annotations

import argparse
import math
import re
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

# 将仓库根目录加入 sys.path，便于复用训练脚本中的组件。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.snn_text_lm import (  # type: ignore
    TOKEN_BOS,
    TOKEN_EOS,
    TextSNNLM,
    Vocab,
    open_stream,
    tokenize,
)
from tools.readability import grade


DEFAULT_CORPUS_GLOBS = ["data/seed_*.txt"]
MAX_LINES_FOR_BUILD = 1200
VOCAB_MAX = 2048
WARMUP_EPOCHS = 1
WARMUP_SEQS_PER_EPOCH = 0
WARMUP_TOKEN_LIMIT = 0
BIGRAM_BLEND = 0.92


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    spike_estimate: float
    readability: float
    context: float
    notes: Sequence[str]


@dataclass
class ContextModel:
    transitions: dict[str, Counter[str]]
    totals: dict[str, int]
    fallback_probs: dict[str, float]


def _gather_sequences(patterns: Sequence[str]) -> List[List[str]]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    sequences: List[List[str]] = []
    for path in paths:
        try:
            for line in open_stream(str(path)):
                line = line.strip()
                if not line:
                    continue
                tokens = tokenize(line)
                if tokens:
                    sequences.append(tokens)
                if len(sequences) >= MAX_LINES_FOR_BUILD:
                    return sequences
        except OSError:
            continue
    if not sequences:
        fallback = "今天天气温和，我们决定在河畔散步。"
        sequences.append(tokenize(fallback))
    return sequences


def _build_vocab(sequences: Sequence[Sequence[str]]) -> Vocab:
    return Vocab.build(sequences, max_size=VOCAB_MAX)


def _warmup_model(model: TextSNNLM, vocab: Vocab, sequences: List[List[str]], seed: int) -> None:
    if WARMUP_SEQS_PER_EPOCH <= 0 or WARMUP_TOKEN_LIMIT <= 0:
        return
    rng = random.Random(seed)
    total_sequences = len(sequences)
    if total_sequences == 0:
        return
    for epoch in range(WARMUP_EPOCHS):
        rng.shuffle(sequences)
        used = 0
        token_budget = 0
        for tokens in sequences:
            seq_tokens = [TOKEN_BOS] + list(tokens) + [TOKEN_EOS]
            model.reset_temporal()
            for idx in range(len(seq_tokens) - 1):
                current = seq_tokens[idx]
                nxt = seq_tokens[idx + 1]
                state = model.forward(current)
                model.update(state, vocab.encode(nxt), train=True)
                token_budget += 1
            used += 1
            if used >= WARMUP_SEQS_PER_EPOCH or token_budget >= WARMUP_TOKEN_LIMIT:
                break


def _build_context_model(sequences: Sequence[Sequence[str]]) -> ContextModel:
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    fallback_counter: Counter[str] = Counter()
    for tokens in sequences:
        prev = TOKEN_BOS
        for token in list(tokens) + [TOKEN_EOS]:
            transitions[prev][token] += 1
            fallback_counter[token] += 1
            prev = token
    totals = {key: sum(counter.values()) for key, counter in transitions.items()}
    fallback_total = sum(fallback_counter.values()) or 1
    fallback_probs = {
        token: count / fallback_total for token, count in fallback_counter.items()
    }
    return ContextModel(transitions=dict(transitions), totals=totals, fallback_probs=fallback_probs)


def _temperature_adjust(probs: Sequence[float], temperature: float) -> List[float]:
    if not probs:
        return []
    if temperature <= 0.0:
        raise ValueError("temperature 必须为正数。")
    if math.isclose(temperature, 1.0, abs_tol=1e-5):
        return list(probs)
    adjusted = [math.pow(max(p, 1e-8), 1.0 / temperature) for p in probs]
    total = sum(adjusted)
    if total <= 0.0:
        return [1.0 / len(adjusted) for _ in adjusted]
    return [val / total for val in adjusted]


def _sample_from_probs(probs: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def _postprocess_text(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("，，", "，").replace("。。", "。").replace("、、", "、")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)
    cleaned = re.sub(r"([。！？])([^。\n！？])", r"\1\n\2", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()
    if cleaned and cleaned[-1] not in "。！？":
        cleaned = f"{cleaned}。"
    return cleaned


def spike_generate(
    *,
    max_len: int = 200,
    seed_text: str = "",
    stop_tokens: Tuple[str, ...] = (TOKEN_EOS,),
    temperature: float = 1.0,
    topic_hint: str | None = None,
    rng_seed: int | None = None,
) -> GenerationResult:
    """基于 TextSNNLM 的尖峰生成循环。"""
    sequences = _gather_sequences(DEFAULT_CORPUS_GLOBS)
    vocab = _build_vocab(sequences)
    model = TextSNNLM(vocab_size=len(vocab.id_to_token))
    _warmup_model(model, vocab, sequences, seed=rng_seed or int(time.time()))
    context_model = _build_context_model(sequences)

    rng = random.Random(rng_seed or int(time.time()))
    model.reset_temporal()

    seed_tokens = tokenize(seed_text) if seed_text else []
    prev_token = TOKEN_BOS
    spike_accum = 0.0
    generated: List[str] = []
    min_generated = max(40, max_len // 4)

    for token in seed_tokens:
        state = model.forward(prev_token)
        spike_accum += sum(state.hidden_rates) * model.inner_steps
        prev_token = token

    for _ in range(max_len):
        state = model.forward(prev_token)
        spike_accum += sum(state.hidden_rates) * model.inner_steps
        probs = list(_temperature_adjust(state.probs, temperature))
        # 避免采样特殊符号
        bos_id = vocab.encode(TOKEN_BOS)
        unk_id = vocab.unk_id
        if 0 <= bos_id < len(probs):
            probs[bos_id] = 0.0
        if 0 <= unk_id < len(probs):
            probs[unk_id] = probs[unk_id] * 0.3
        eos_id = vocab.encode(TOKEN_EOS)
        if len(generated) < min_generated and 0 <= eos_id < len(probs):
            probs[eos_id] = 0.0
        if context_model.transitions:
            counts = context_model.transitions.get(prev_token)
            blended = [p * (1.0 - BIGRAM_BLEND) for p in probs]
            if counts:
                total_counts = context_model.totals.get(prev_token, 0)
                if total_counts > 0:
                    for token, cnt in counts.items():
                        idx = vocab.token_to_id.get(token)
                        if idx is None or idx < 0 or idx >= len(blended):
                            continue
                        blended[idx] += BIGRAM_BLEND * (cnt / total_counts)
            else:
                for token, prob in context_model.fallback_probs.items():
                    idx = vocab.token_to_id.get(token)
                    if idx is None or idx < 0 or idx >= len(blended):
                        continue
                    blended[idx] += BIGRAM_BLEND * prob
            probs = blended
        if len(generated) < min_generated and 0 <= eos_id < len(probs):
            probs[eos_id] = 0.0

        total = sum(probs)
        if total <= 0.0:
            probs = [1.0 / len(probs) for _ in probs]
        else:
            probs = [p / total for p in probs]
        sample_id = _sample_from_probs(probs, rng)
        token = vocab.id_to_token[sample_id]
        if token in stop_tokens:
            break
        generated.append(token)
        prev_token = token

    topic_prefix = ""
    if topic_hint:
        topic_prefix = f"主题：{topic_hint}\n"
    generated_text = "".join(generated)
    base_text = _postprocess_text(f"{seed_text}{generated_text}")
    full_text = f"{topic_prefix}{base_text}" if topic_prefix else base_text
    evaluation = grade(full_text, topic_hint=topic_hint)

    return GenerationResult(
        text=full_text,
        tokens_generated=len(generated),
        spike_estimate=spike_accum,
        readability=evaluation["readability"],
        context=evaluation["context"],
        notes=evaluation["notes"],
    )


def _write_feed(result: GenerationResult, topic_hint: str | None, max_len: int, temperature: float) -> Path:
    runs_dir = Path("runs/feed")
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = runs_dir / f"{timestamp}.md"
    notes_text = "\n".join(f"- {note}" for note in result.notes)
    metadata = (
        f"# Spike Writer Output\n"
        f"- Timestamp: {timestamp}\n"
        f"- Topic: {topic_hint or '未指定'}\n"
        f"- Max Length: {max_len}\n"
        f"- Temperature: {temperature:.2f}\n"
        f"- Generated Tokens: {result.tokens_generated}\n"
        f"- Spike Estimate: {result.spike_estimate:.2f}\n"
        f"- Readability: {result.readability:.4f}\n"
        f"- Context Alignment: {result.context:.4f}\n"
        f"- Notes:\n{notes_text}\n\n"
        f"## Content\n"
        f"{result.text}\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(metadata)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="尖峰神经网络文本生成器")
    parser.add_argument("--len", type=int, default=200, help="最大生成字符数。")
    parser.add_argument("--seed", type=str, default="", help="注入的起始文本。")
    parser.add_argument("--topic", type=str, default="", help="主题提示，将写入前缀。")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度。")
    parser.add_argument("--stop-token", action="append", default=[TOKEN_EOS], help="终止 token。")
    parser.add_argument("--rng-seed", type=int, default=None, help="随机种子，便于复现。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stop_tokens = tuple(args.stop_token) if args.stop_token else (TOKEN_EOS,)
    result = spike_generate(
        max_len=args.len,
        seed_text=args.seed,
        stop_tokens=stop_tokens,
        temperature=args.temperature,
        topic_hint=args.topic or None,
        rng_seed=args.rng_seed,
    )
    feed_path = _write_feed(result, args.topic or None, args.len, args.temperature)
    print(f"[writer] feed saved to {feed_path}")


if __name__ == "__main__":
    main()
