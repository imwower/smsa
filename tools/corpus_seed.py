"""Generate lightweight, readable synthetic corpora (standard library only).

Provides two helpers used by README and daemon workflows:
- random_topics(): deterministic list of topic labels
- write_corpus(path, lines, topic_hint): write semi-readable Chinese lines

No external dependencies.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List


_TOPICS = [
    "出行", "职场", "科技", "烹饪", "教育", "医疗",
    "体育", "游戏", "旅游", "生活", "金融", "文学",
]


def random_topics(seed: int = 2025) -> List[str]:
    rng = random.Random(seed)
    topics = list(_TOPICS)
    rng.shuffle(topics)
    return topics


def _lexicon(topic: str) -> List[str]:
    base = [
        "今天", "我们", "可以", "尝试", "理解", "如何", "通过",
        "简单", "方法", "逐步", "提升", "系统", "能力",
        "同时", "关注", "能耗", "稳定", "数据", "指标",
    ]
    extra = {
        "出行": ["路线", "地铁", "站点", "步行", "换乘", "出发", "到达"],
        "职场": ["会议", "协作", "复盘", "节奏", "计划", "汇报", "成长"],
        "科技": ["计算", "模型", "算法", "平台", "部署", "接口", "数据"],
        "烹饪": ["食材", "火候", "香味", "锅铲", "米饭", "汤汁", "口感"],
        "教育": ["课堂", "练习", "思考", "反馈", "作业", "启发", "专注"],
        "医疗": ["诊断", "指标", "体征", "治疗", "护理", "恢复", "健康"],
        "体育": ["训练", "速度", "配合", "体能", "比赛", "战术", "目标"],
        "游戏": ["关卡", "技能", "资源", "升级", "队友", "策略", "平衡"],
        "旅游": ["风景", "路线", "酒店", "美食", "拍照", "地图", "步道"],
        "生活": ["阳光", "早餐", "清单", "家务", "心情", "节奏", "休息"],
        "金融": ["账本", "预算", "收益", "风险", "资产", "花费", "分配"],
        "文学": ["诗句", "叙事", "隐喻", "篇章", "意象", "韵律", "题旨"],
    }.get(topic, [])
    return base + extra


def _sentences(topic: str, length: int, rng: random.Random) -> Iterable[str]:
    words = _lexicon(topic)
    n = max(6, min(18, length // 10))
    punct = ["。", "！", "？"]
    for _ in range(length):
        k = rng.randint(max(6, n - 2), n + 2)
        tokens = [rng.choice(words) for _ in range(k)]
        # add a small structure
        if rng.random() < 0.3:
            tokens.insert(0, topic)
        sent = "".join(tokens) + rng.choice(punct)
        yield sent


def write_corpus(path: str | Path, lines: int = 500, topic_hint: str | None = None, seed: int | None = None) -> Path:
    """Write a small text corpus with semi-readable lines.

    - path: output file path (parent dirs are created)
    - lines: number of lines to write
    - topic_hint: one of random_topics() (used to bias lexicon)
    - seed: optional RNG seed for reproducibility
    """
    rng = random.Random(seed if seed is not None else 2027)
    topic = topic_hint or random_topics(rng.randrange(10000))[0]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for s in _sentences(topic, lines, rng):
            fh.write(s + "\n")
    return out


__all__ = ["write_corpus", "random_topics"]

