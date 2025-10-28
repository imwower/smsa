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
from tools.explainability import explainability_index


DEFAULT_CORPUS_GLOBS = ["data/seed_*.txt"]
MAX_LINES_FOR_BUILD = 1200
VOCAB_MAX = 2048
WARMUP_EPOCHS = 1
WARMUP_SEQS_PER_EPOCH = 0
WARMUP_TOKEN_LIMIT = 0
BIGRAM_BLEND = 0.92

# AUTOPATCH DECODE PARAMS START
# 默认解码参数（可由 AutoPatch 在锚点内调整）
DECODE_TOP_K = 50
DECODE_REPEAT_PENALTY = 1.1
# AUTOPATCH DECODE PARAMS END

# 运行时解码上下文（用于重复惩罚与 trigram 阻断）
_DECODE_HISTORY_IDS: list[int] = []
_DECODE_TRIGRAMS: set[tuple[int, int, int]] = set()

def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    apex = max(logits)
    exps = [math.exp(val - apex) for val in logits]
    denom = sum(exps)
    if denom <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [v / denom for v in exps]


def sample_token(
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
    top_k: int = DECODE_TOP_K,
    repeat_penalty: float = DECODE_REPEAT_PENALTY,
    trigram_block: bool = True,
) -> int:
    """标准库实现的采样器（top‑k / 温度 / 重复惩罚 / trigram 阻断）。

    说明：
    - logits 可为任意实数分数（相对大小决定选择概率）
    - 温度通过对 logits 除以 temperature 实现；temperature→0 趋于贪心
    - top_k 仅保留最高 K 个分数；其余设为 -inf
    - 重复惩罚：若 token 历史频次为 c，则分数 / (repeat_penalty ** c)
    - trigram 阻断：若最近两 token 与候选构成的三元组出现过，则屏蔽
    """
    global _DECODE_HISTORY_IDS, _DECODE_TRIGRAMS
    if temperature <= 0.0:
        raise ValueError("temperature 必须为正数")

    scores = list(logits)
    # 温度缩放
    if not math.isclose(temperature, 1.0, abs_tol=1e-6):
        inv = 1.0 / temperature
        scores = [val * inv for val in scores]

    # top-k 过滤
    k = max(1, min(int(top_k), len(scores)))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    mask = [False] * len(scores)
    for i in top_idx:
        mask[i] = True
    min_val = min(scores) - 1e6
    scores = [scores[i] if mask[i] else min_val for i in range(len(scores))]

    # 重复惩罚
    if repeat_penalty > 1.0 and _DECODE_HISTORY_IDS:
        freq = Counter(_DECODE_HISTORY_IDS)
        for i in range(len(scores)):
            c = freq.get(i, 0)
            if c > 0:
                scores[i] = scores[i] / (repeat_penalty ** c)

    # trigram 阻断（仅当历史长度 ≥2）
    if trigram_block and len(_DECODE_HISTORY_IDS) >= 2 and _DECODE_TRIGRAMS:
        a, b = _DECODE_HISTORY_IDS[-2], _DECODE_HISTORY_IDS[-1]
        for i in range(len(scores)):
            if (a, b, i) in _DECODE_TRIGRAMS:
                scores[i] = min_val

    # 采样
    probs = _softmax(scores)
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


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
        # 计算 logits（基于隐藏率和读出头），便于进行 top-k / 温度 / 惩罚
        logits = []
        for vidx in range(len(model.readout_bias)):
            val = model.readout_bias[vidx]
            for h in range(model.hidden.n_out):
                val += model.readout_weights[h][vidx] * state.hidden_rates[h]
            logits.append(val)

        # 避免特殊 token（BOS/UNK 提前抑制；在短序列阶段禁用 EOS）
        bos_id = vocab.encode(TOKEN_BOS)
        unk_id = vocab.unk_id
        eos_id = vocab.encode(TOKEN_EOS)
        if 0 <= bos_id < len(logits):
            logits[bos_id] = -1e9
        if 0 <= unk_id < len(logits):
            logits[unk_id] -= 2.0
        if len(generated) < min_generated and 0 <= eos_id < len(logits):
            logits[eos_id] = -1e9

        # Bigram 融合：作为先验加成（转为 logits 空间近似相加）
        if context_model.transitions:
            counts = context_model.transitions.get(prev_token)
            if counts:
                total_counts = context_model.totals.get(prev_token, 0)
                if total_counts > 0:
                    for tok, cnt in counts.items():
                        idx = vocab.token_to_id.get(tok)
                        if idx is None or idx < 0 or idx >= len(logits):
                            continue
                        logits[idx] += math.log(1e-8 + BIGRAM_BLEND * (cnt / total_counts))
            else:
                for tok, prob in context_model.fallback_probs.items():
                    idx = vocab.token_to_id.get(tok)
                    if idx is None or idx < 0 or idx >= len(logits):
                        continue
                    logits[idx] += math.log(1e-8 + BIGRAM_BLEND * prob)

        # AUTOPATCH DECODE LOGIC START
        # 可由 AutoPatch 切换 trigram 阻断/采样策略
        sample_id = sample_token(
            logits,
            temperature=temperature,
            top_k=DECODE_TOP_K,
            repeat_penalty=DECODE_REPEAT_PENALTY,
            trigram_block=True,
        )
        # AUTOPATCH DECODE LOGIC END

        token = vocab.id_to_token[sample_id]
        if token in stop_tokens:
            break
        generated.append(token)
        prev_token = token
        # 更新解码上下文（历史与 trigram 集）
        _DECODE_HISTORY_IDS.append(sample_id)
        if len(_DECODE_HISTORY_IDS) >= 3:
            tri = tuple(_DECODE_HISTORY_IDS[-3:])
            _DECODE_TRIGRAMS.add(tri)  # type: ignore[arg-type]

    topic_prefix = ""
    if topic_hint:
        topic_prefix = f"主题：{topic_hint}\n"
    generated_text = "".join(generated)
    base_text = _postprocess_text(f"{seed_text}{generated_text}")
    full_text = f"{topic_prefix}{base_text}" if topic_prefix else base_text
    # Explainability 评分（替代原 readability.grade）
    ei = explainability_index(full_text, topic_hint)

    return GenerationResult(
        text=full_text,
        tokens_generated=len(generated),
        spike_estimate=spike_accum,
        readability=float(ei.get("readability", 0.0)),
        context=float(ei.get("context", 0.0)),
        notes=list(ei.get("notes", [])),
    )


def _write_feed(result: GenerationResult, topic_hint: str | None, max_len: int, temperature: float) -> Path:
    runs_dir = Path("runs/feed")
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = runs_dir / f"{timestamp}.md"
    # Explainability 额外字段
    ei = explainability_index(result.text, topic_hint)
    notes_text = "\n".join(f"- {note}" for note in (ei.get("notes") or []))
    metadata = (
        f"# Spike Writer Output\n"
        f"- Timestamp: {timestamp}\n"
        f"- Topic: {topic_hint or '未指定'}\n"
        f"- Max Length: {max_len}\n"
        f"- Temperature: {temperature:.2f}\n"
        f"- Generated Tokens: {result.tokens_generated}\n"
        f"- Spike Estimate: {result.spike_estimate:.2f}\n"
        f"- Readability: {ei.get('readability', 0.0):.4f}\n"
        f"- Context Alignment: {ei.get('context', 0.0):.4f}\n"
        f"- Self-Explain: {ei.get('self_explain', 0.0):.4f}\n"
        f"- Overall Index: {ei.get('overall', 0.0):.4f}\n"
        f"- Notes:\n{notes_text}\n\n"
        f"## Content\n"
        f"{result.text}\n\n"
        f"## Explainability\n"
        f"readability={ei.get('readability', 0.0):.4f} "
        f"context={ei.get('context', 0.0):.4f} "
        f"self_explain={ei.get('self_explain', 0.0):.4f} "
        f"overall={ei.get('overall', 0.0):.4f}\n"
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
