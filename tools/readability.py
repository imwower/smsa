"""轻量可读性与语境评估工具，仅依赖标准库。"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence


__all__ = [
    "readability_score",
    "context_score",
    "overall_score",
    "grade",
]


CHINESE_PUNCTUATION = {"，", "。", "！", "？", "；", "：", "、", "《", "》", "“", "”", "（", "）", "——"}
ASCII_PUNCTUATION = {",", ".", "!", "?", ";", ":", "'", '"', "(", ")", "-", "–"}
PUNCT_MAP = {
    "comma": {",", "，"},
    "period": {".", "。", "!", "！", "?", "？", ";", "；", ":"},
    "dunhao": {"、"},
}


def readability_score(text: str, lang: str = "zh") -> float:
    """计算综合可读性得分，范围 0~1。"""
    text = text or ""
    stripped = text.strip()
    if not stripped:
        return 0.0

    tokens = _tokenize(stripped, lang)
    sentences = _split_sentences(stripped)
    sentence_lengths = [len(_tokenize(sentence, lang)) for sentence in sentences if sentence]

    scores: Dict[str, float] = {}
    scores["coverage"] = _character_coverage(stripped)
    scores["sentence_mean"] = _sentence_mean_score(sentence_lengths, lang)
    scores["sentence_var"] = _sentence_variance_score(sentence_lengths)
    scores["repetition"] = _trigram_diversity(tokens)
    scores["lexical"] = _lexical_diversity(tokens)
    scores["rhythm"] = _punctuation_rhythm(stripped, tokens, lang)

    weights = {
        "coverage": 0.15,
        "sentence_mean": 0.2,
        "sentence_var": 0.1,
        "repetition": 0.2,
        "lexical": 0.2,
        "rhythm": 0.15,
    }
    total_weight = sum(weights.values())
    blended = sum(scores[name] * weights[name] for name in scores) / total_weight
    blended = _apply_length_factor(blended, len(tokens))
    return max(0.0, min(1.0, blended))


def context_score(text: str, topic_hint: str | None) -> float:
    """根据主题提示估算语境匹配程度，未提供提示时返回 0.5。"""
    if topic_hint is None or not topic_hint.strip():
        return 0.5

    lang = "zh" if _contains_chinese(topic_hint) else "en"
    text_tokens = set(token.lower() for token in _tokenize(text, lang) if token)
    keywords = [kw.lower() for kw in _tokenize(topic_hint, lang) if kw]
    if not keywords or not text_tokens:
        return 0.2 if keywords else 0.5

    hits = sum(1 for kw in keywords if kw in text_tokens)
    coverage = hits / len(keywords)
    density = hits / (len(text_tokens) or 1)
    score = 0.6 * coverage + 0.4 * min(density * 3.0, 1.0)
    return max(0.0, min(1.0, score))


def overall_score(text: str, topic_hint: str | None = None, lang: str | None = None) -> float:
    """组合可读性与语境得分，默认按 0.7/0.3 加权。"""
    lang = lang or ("zh" if _contains_chinese(text) else "en")
    readability = readability_score(text, lang=lang)
    context = context_score(text, topic_hint)
    combined = 0.7 * readability + 0.3 * context
    return max(0.0, min(1.0, combined))


def grade(text: str, topic_hint: str | None = None) -> Dict[str, object]:
    """汇总评分结果并附带说明。"""
    lang = "zh" if _contains_chinese(text) else "en"
    readability = readability_score(text, lang=lang)
    context = context_score(text, topic_hint)
    overall = 0.7 * readability + 0.3 * context

    notes: List[str] = []
    if readability < 0.4:
        notes.append("可读性偏低，建议优化句式或减少重复。")
    if context < 0.5 and topic_hint:
        notes.append("主题命中率不足，建议补充相关关键词。")
    if not notes:
        notes.append("文本表现稳定，无明显风险。")

    return {
        "readability": round(readability, 4),
        "context": round(context, 4),
        "overall": round(overall, 4),
        "notes": notes,
    }


def _character_coverage(text: str) -> float:
    if not text:
        return 0.0
    allowed_chars = 0
    language_chars = 0
    punctuation_chars = 0
    digit_chars = 0
    total = len(text)
    for ch in text:
        if ch.isspace():
            allowed_chars += 1
            continue
        if _is_chinese(ch) or ch.isalpha():
            allowed_chars += 1
            language_chars += 1
            continue
        if ch.isdigit():
            allowed_chars += 1
            digit_chars += 1
            continue
        if ch in CHINESE_PUNCTUATION or ch in ASCII_PUNCTUATION:
            allowed_chars += 1
            punctuation_chars += 1
    coverage = allowed_chars / total
    language_ratio = language_chars / total
    symbol_ratio = (punctuation_chars + digit_chars) / total
    penalty = max(0.0, symbol_ratio - 0.35)
    score = coverage * (0.6 + 0.4 * language_ratio)
    score *= max(0.0, 1.0 - penalty * 1.3)
    return max(0.0, min(1.0, score))


def _sentence_mean_score(lengths: Sequence[int], lang: str) -> float:
    if not lengths:
        return 0.0
    mean_len = sum(lengths) / len(lengths)
    ideal = 20.0 if lang.startswith("zh") else 14.0
    tolerance = ideal * 0.6
    delta = mean_len - ideal
    score = math.exp(-0.5 * (delta / (tolerance or 1.0)) ** 2)
    return max(0.0, min(1.0, score))


def _sentence_variance_score(lengths: Sequence[int]) -> float:
    if not lengths:
        return 0.0
    mean_len = sum(lengths) / len(lengths)
    variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
    score = math.exp(-variance / (mean_len * mean_len + 1e-6))
    return max(0.0, min(1.0, score))


def _trigram_diversity(tokens: Sequence[str]) -> float:
    if len(tokens) < 3:
        return 1.0
    triples = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(triples)
    total = len(triples)
    repeats = sum(count - 1 for count in counts.values())
    penalty = repeats / total
    return max(0.0, 1.0 - penalty)


def _lexical_diversity(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    normalized = [token.lower() for token in tokens if token]
    unique_tokens = len(set(normalized))
    diversity = unique_tokens / len(normalized)
    smoothed = (diversity * len(normalized)) / (len(normalized) + 4.0)
    return max(0.0, min(1.0, smoothed))


def _punctuation_rhythm(text: str, tokens: Sequence[str], lang: str) -> float:
    counts = {bucket: 0 for bucket in PUNCT_MAP}
    for bucket, marks in PUNCT_MAP.items():
        counts[bucket] = sum(text.count(mark) for mark in marks)
    total = sum(counts.values())
    if total == 0:
        return 0.35

    target = {"comma": 0.5, "period": 0.4, "dunhao": 0.1}
    actual = {bucket: counts[bucket] / total for bucket in counts}
    diff = sum(abs(actual[bucket] - target[bucket]) for bucket in target) / 2.0
    balance_score = max(0.0, 1.0 - diff * 1.6)

    density = total / max(len(tokens), 1)
    ideal_density = 0.05 if lang.startswith("zh") else 0.1
    density_score = math.exp(-abs(density - ideal_density) / (ideal_density + 1e-6))

    combined = 0.6 * balance_score + 0.4 * density_score
    return max(0.0, min(1.0, combined))


def _tokenize(text: str, lang: str) -> List[str]:
    if not text:
        return []
    if lang.startswith("zh") or _contains_chinese(text):
        tokens: List[str] = []
        buffer = []
        for ch in text:
            if _is_chinese(ch):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(ch)
            elif ch.isalnum():
                buffer.append(ch.lower())
            else:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
        if buffer:
            tokens.append("".join(buffer))
        return tokens
    return [token.lower() for token in re.split(r"\s+", text) if token]


def _split_sentences(text: str) -> List[str]:
    pieces = re.split(r"[。！？!?\.]+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _is_chinese(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _contains_chinese(text: str) -> bool:
    return any(_is_chinese(ch) for ch in text or "")


def _apply_length_factor(score: float, token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    if token_count < 6:
        factor = 0.2 + 0.8 * (token_count / 6.0)
    elif token_count < 12:
        factor = 0.6 + 0.4 * (token_count / 12.0)
    else:
        factor = 1.0
    return score * max(0.0, min(1.0, factor))


if __name__ == "__main__":
    # 最小用例：快速检验接口行为。
    demo = "今天天气不错，我们计划在公园里散步，然后讨论项目细节。"
    print(grade(demo))
