"""基于尖峰神经网络的文本生成器，复用 TextSNNLM 读出模块进行采样。

增强项：
- 若未提供 seed_text，自动注入 <bos> 或基于 topic 的默认提示；
- 设定最小约束 MIN_TOKENS/MIN_SPIKES，并在不达标时按三组解码参数依次重试；
- 每次尝试后记录 spikes/tokens/params 与 explainability 指标；
- 若三次仍不达标，回退到“兜底文段”（高频 token 采样生成 ≥80 字，并标注为“兜底”）。
"""

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

# 生成最小约束
MIN_TOKENS = 80
MIN_SPIKES = 20.0

# AUTOPATCH DECODE PARAMS START
# 默认解码参数（可由 AutoPatch 在锚点内调整)
DECODE_TOP_K = 80

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
class AttemptLog:
    attempt: int
    temperature: float
    top_k: int
    repeat_penalty: float
    trigram_block: bool
    tokens: int
    spikes: float
    readability: float
    context: float
    self_explain: float
    overall: float
    passed: bool


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    spike_estimate: float
    readability: float
    context: float
    notes: Sequence[str]
    attempts: List[AttemptLog]
    fallback_used: bool = False


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
    top_k: int = DECODE_TOP_K,
    repeat_penalty: float = DECODE_REPEAT_PENALTY,
    trigram_block: bool = True,
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
    # 加强最小长度：至少满足 MIN_TOKENS
    min_generated = max(MIN_TOKENS, max_len // 4)

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
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            trigram_block=trigram_block,
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
        attempts=[],
        fallback_used=False,
    )


def _default_seed(topic_hint: str | None) -> str:
    if topic_hint:
        return f"围绕“{topic_hint}”，我们以学习与探索为线索，提出问题、举例推演、再归纳洞见。"
    # 无主题时保持简短提示，由 <bos> 引导内部状态
    return "从感知出发，逐步连接事实与观点。"


def _fallback_paragraph(context_model: ContextModel, vocab: Vocab, *, min_chars: int = 80, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    # 按语料全局频率采样，屏蔽 BOS/EOS
    probs = context_model.fallback_probs or {}
    items = [(tok, p) for tok, p in probs.items() if tok not in (TOKEN_BOS, TOKEN_EOS)]
    if not items:
        return "【兜底】在主题的脉络里，我们用简洁的语言完成一段可读的表述。"
    tokens, weights = zip(*items)
    # 归一化
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    text = []
    while len("".join(text)) < min_chars:
        idx = _sample_from_probs(weights, rng)
        text.append(tokens[idx])
    paragraph = "".join(text)
    paragraph = _postprocess_text(paragraph)
    return f"【兜底】{paragraph}"


def _evaluate_attempt(result: GenerationResult, *, attempt_idx: int, temperature: float, top_k: int, repeat_penalty: float, trigram_block: bool, topic_hint: str | None) -> AttemptLog:
    ei = explainability_index(result.text, topic_hint)
    return AttemptLog(
        attempt=attempt_idx,
        temperature=temperature,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        trigram_block=trigram_block,
        tokens=result.tokens_generated,
        spikes=result.spike_estimate,
        readability=float(ei.get("readability", 0.0)),
        context=float(ei.get("context", 0.0)),
        self_explain=float(ei.get("self_explain", 0.0)),
        overall=float(ei.get("overall", 0.0)),
        passed=(result.tokens_generated >= MIN_TOKENS and result.spike_estimate >= MIN_SPIKES),
    )


def run_with_retries(
    *,
    max_len: int,
    seed_text: str,
    stop_tokens: Tuple[str, ...],
    temperature: float,
    topic_hint: str | None,
    rng_seed: int | None,
) -> GenerationResult:
    # 若未提供 seed_text，自动注入 <bos> 或默认提示（含主题）
    if not seed_text:
        seed_text = _default_seed(topic_hint)

    attempts: List[AttemptLog] = []

    # 尝试 0：使用用户给定温度与默认 top_k/penalty
    base = spike_generate(
        max_len=max_len,
        seed_text=seed_text,
        stop_tokens=stop_tokens,
        temperature=temperature,
        top_k=DECODE_TOP_K,
        repeat_penalty=DECODE_REPEAT_PENALTY,
        trigram_block=True,
        topic_hint=topic_hint,
        rng_seed=rng_seed,
    )
    attempts.append(_evaluate_attempt(base, attempt_idx=0, temperature=temperature, top_k=DECODE_TOP_K, repeat_penalty=DECODE_REPEAT_PENALTY, trigram_block=True, topic_hint=topic_hint))
    if base.tokens_generated >= MIN_TOKENS and base.spike_estimate >= MIN_SPIKES:
        base.attempts = attempts
        return base

    # 尝试 1（规范参数组 1）
    t1, k1, rp1, tb1 = 0.90, 48, 1.10, False
    _DECODE_HISTORY_IDS.clear(); _DECODE_TRIGRAMS.clear()
    a1 = spike_generate(
        max_len=max_len,
        seed_text=seed_text,
        stop_tokens=stop_tokens,
        temperature=t1,
        top_k=k1,
        repeat_penalty=rp1,
        trigram_block=tb1,
        topic_hint=topic_hint,
        rng_seed=(rng_seed + 1) if rng_seed is not None else None,
    )
    attempts.append(_evaluate_attempt(a1, attempt_idx=1, temperature=t1, top_k=k1, repeat_penalty=rp1, trigram_block=tb1, topic_hint=topic_hint))
    if a1.tokens_generated >= MIN_TOKENS and a1.spike_estimate >= MIN_SPIKES:
        a1.attempts = attempts
        return a1

    # 尝试 2（规范参数组 2）
    t2, k2, rp2, tb2 = 0.85, 64, 1.12, True
    _DECODE_HISTORY_IDS.clear(); _DECODE_TRIGRAMS.clear()
    a2 = spike_generate(
        max_len=max_len,
        seed_text=seed_text,
        stop_tokens=stop_tokens,
        temperature=t2,
        top_k=k2,
        repeat_penalty=rp2,
        trigram_block=tb2,
        topic_hint=topic_hint,
        rng_seed=(rng_seed + 2) if rng_seed is not None else None,
    )
    attempts.append(_evaluate_attempt(a2, attempt_idx=2, temperature=t2, top_k=k2, repeat_penalty=rp2, trigram_block=tb2, topic_hint=topic_hint))
    if a2.tokens_generated >= MIN_TOKENS and a2.spike_estimate >= MIN_SPIKES:
        a2.attempts = attempts
        return a2

    # 尝试 3（规范参数组 3）
    t3, k3, rp3, tb3 = 0.80, 72, 1.15, True
    _DECODE_HISTORY_IDS.clear(); _DECODE_TRIGRAMS.clear()
    a3 = spike_generate(
        max_len=max_len,
        seed_text=seed_text,
        stop_tokens=stop_tokens,
        temperature=t3,
        top_k=k3,
        repeat_penalty=rp3,
        trigram_block=tb3,
        topic_hint=topic_hint,
        rng_seed=(rng_seed + 3) if rng_seed is not None else None,
    )
    attempts.append(_evaluate_attempt(a3, attempt_idx=3, temperature=t3, top_k=k3, repeat_penalty=rp3, trigram_block=tb3, topic_hint=topic_hint))
    if a3.tokens_generated >= MIN_TOKENS and a3.spike_estimate >= MIN_SPIKES:
        a3.attempts = attempts
        return a3

    # 三次仍不达标：兜底文段（≥80 字，文末标注 [FALLBACK]）
    # 复用最后一次的上下文模型构造逻辑（用 a3 的评分），这里重新构建以取 fallback 概率
    sequences = _gather_sequences(DEFAULT_CORPUS_GLOBS)
    vocab = _build_vocab(sequences)
    context_model = _build_context_model(sequences)
    fallback_text = _fallback_paragraph(context_model, vocab, min_chars=MIN_TOKENS)
    # 拼接主题前缀
    topic_prefix = f"主题：{topic_hint}\n" if topic_hint else ""
    final_text = f"{topic_prefix}{fallback_text} [FALLBACK]" if topic_prefix else f"{fallback_text} [FALLBACK]"

    # 汇总：以 a3 为基准，替换文本与计数
    a3.text = final_text
    a3.tokens_generated = max(MIN_TOKENS, a3.tokens_generated)
    a3.spike_estimate = max(0.0, a3.spike_estimate)
    a3.attempts = attempts
    a3.fallback_used = True
    return a3


def _write_feed(result: GenerationResult, topic_hint: str | None, max_len: int, temperature: float) -> Path:
    runs_dir = Path("runs/feed")
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = runs_dir / f"{timestamp}.md"
    # Explainability 额外字段
    ei = explainability_index(result.text, topic_hint)
    notes_text = "\n".join(f"- {note}" for note in (ei.get("notes") or []))
    # 追加每次重试的参数与评分
    attempts_lines: List[str] = []
    for log in result.attempts:
        attempts_lines.append(
            " | ".join(
                [
                    f"[try#{log.attempt}]",
                    f"T={log.temperature:.2f}",
                    f"top_k={log.top_k}",
                    f"repeat_penalty={log.repeat_penalty:.2f}",
                    f"trigram={int(log.trigram_block)}",
                    f"tokens={log.tokens}",
                    f"spikes={log.spikes:.2f}",
                    f"read={log.readability:.4f}",
                    f"ctx={log.context:.4f}",
                    f"self={log.self_explain:.4f}",
                    f"overall={log.overall:.4f}",
                    f"pass={int(log.passed)}",
                ]
            )
        )

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
        f"- Fallback Used: {int(result.fallback_used)}\n"
        f"- Attempts:\n{chr(10).join('  - ' + l for l in attempts_lines) if attempts_lines else '  - (none)'}\n\n"
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
    result = run_with_retries(
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
