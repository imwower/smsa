"""Seed corpus generator (Chinese templates + synonym perturbations).

Only uses the standard library. Produces readable short sentences that are
useful for bootstrapping local language-model training or daemon loops.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

__all__ = ["generate_sentences", "write_corpus", "random_topics"]


_SUBJECTS = [
    "我们",
    "团队",
    "老师",
    "同学",
    "研究者",
    "工程师",
    "读者",
    "系统",
    "模型",
]

_ACTIONS = [
    ["讨论", "交流", "探讨"],
    ["观察", "记录", "留意"],
    ["理解", "把握", "掌握"],
    ["分析", "研究", "评估"],
    ["测试", "验证", "检查"],
    ["改进", "优化", "调整"],
    ["尝试", "实践", "演练"],
]

_OBJECTS = [
    ["问题", "现象", "细节"],
    ["计划", "方案", "路径"],
    ["模型", "系统", "程序"],
    ["习惯", "流程", "步骤"],
    ["数据", "样本", "记录"],
    ["训练", "学习", "推理"],
]

_ADVERBS = [
    ["逐步", "缓慢", "稳妥"],
    ["快速", "及时", "立即"],
    ["进一步", "持续", "反复"],
]

_CONNECTIVES = [
    ["因为", "由于"],
    ["所以", "因此"],
    ["于是", "从而"],
]

_RESULTS = [
    ["提升了效果", "改善了表现", "降低了误差"],
    ["稳定了训练", "加快了收敛", "减少了抖动"],
    ["明确了方向", "澄清了疑问", "聚焦了目标"],
]

_ENDINGS = ["。", "。", "。", "！", "。"]


def _choice(rng: random.Random, items: Sequence[str]) -> str:
    return items[rng.randrange(len(items))]


def _choice_group(rng: random.Random, groups: Sequence[Sequence[str]]) -> str:
    return _choice(rng, _choice(rng, groups))


def random_topics() -> List[str]:
    """Return a deterministic list of generic Chinese topic labels."""
    return [
        "science", "life", "news", "sports", "tech",
        "finance", "education", "health", "travel", "culture",
    ]


def generate_sentences(
    n: int,
    *,
    seed: int | None = None,
    topic_hint: str | None = None,
) -> List[str]:
    """Generate n short Chinese sentences with mild variability.

    The generator stitches together simple subject–verb–object clauses, with
    connectors and result phrases to keep them readable and diverse.
    """
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(max(0, n)):
        s = _choice(rng, _SUBJECTS)
        adv = _choice_group(rng, _ADVERBS)
        v = _choice_group(rng, _ACTIONS)
        o = _choice_group(rng, _OBJECTS)
        c1 = _choice_group(rng, _CONNECTIVES)
        c2 = _choice_group(rng, _CONNECTIVES)
        r = _choice_group(rng, _RESULTS)
        # Two-clause template with causal connective
        # e.g., 我们逐步分析数据，因为提出了新方案，因此提升了效果。
        topic_token = f"（{topic_hint}）" if topic_hint else ""
        parts = [
            s,
            adv,
            v,
            o,
            "，",
            c1,
            _choice(rng, ["我们", "大家", s]),
            _choice_group(rng, _ACTIONS),
            _choice_group(rng, _OBJECTS),
            "，",
            c2,
            r,
            topic_token,
        ]
        text = "".join(parts) + _choice(rng, _ENDINGS)
        # Minor punctuation cleanup / style tweak
        text = text.replace("，，", "，").replace("、、", "、")
        lines.append(text)
    return lines


def write_corpus(
    path: str | Path,
    num_lines: int | None = None,
    *,
    lines: int | None = None,
    seed: int | None = None,
    topic_hint: str | None = None,
) -> Path:
    """Write a plain-text corpus file with the requested number of lines.

    Returns the absolute Path to the created file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total = int(lines if lines is not None else (num_lines if num_lines is not None else 0))
    if total <= 0:
        total = 800
    seq = generate_sentences(total, seed=seed, topic_hint=topic_hint)
    with p.open("w", encoding="utf-8") as fh:
        for line in seq:
            fh.write(line)
            fh.write("\n")
    return p.resolve()
