"""Synthetic Chinese corpus seeding utilities (template + synonym noise).

Standard-library only. Generates readable short sentences with topic-focused
vocabulary. Intended for bootstrapping the LM daemon with a few seed files.

Functions
- generate_sentences(count, topic, seed): return a list of sentences.
- write_corpus(output_dir="data", topics=("news","science","life"),
               lines_per_file=800, seed=None): write data/seed_*.txt files.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = ["generate_sentences", "write_corpus", "random_topics"]


# Core synonym pools organized by topic. Light, readable vocabulary that
# composes well across templates.
_COMMON: Mapping[str, Sequence[str]] = {
    "time": ("今天", "昨日", "近期", "本周", "早晨", "傍晚", "午后"),
    "link": ("因此", "从而", "于是", "所以", "最终", "随后"),
    "modal": ("正在", "将会", "已经", "仍在", "准备"),
    "freq": ("经常", "偶尔", "持续", "逐步", "不断", "稳定"),
    "degree": ("显著", "明显", "稳健", "细微", "快速", "温和"),
    "place": (
        "本地", "校园", "社区", "公司", "实验室", "车站", "图书馆", "公园", "中心"
    ),
    "aux": ("进一步", "全面", "有效", "及时", "正式", "积极"),
}

_TOPICS: Mapping[str, Mapping[str, Sequence[str]]] = {
    "news": {
        "subj": ("部门", "警方", "学校", "企业", "社区", "志愿者", "团队"),
        "verb": ("发布", "宣布", "推进", "启动", "升级", "优化", "落实"),
        "obj": (
            "新政策", "服务措施", "安全计划", "合作项目", "便民活动", "援助方案", "培训课程"
        ),
        "tail": (
            "以提升公众体验", "保障市民安全", "促进秩序改善", "带动区域发展", "回应社会关切",
            "改善办事流程",
        ),
    },
    "science": {
        "subj": ("研究团队", "科学家", "工程师", "学者", "实验小组", "天文学家"),
        "verb": ("发现", "验证", "提出", "改进", "复现", "揭示"),
        "obj": (
            "新模型", "算法原型", "观测结果", "实验方法", "关键机制", "性能上限"
        ),
        "tail": (
            "在多项评测中表现{degree}", "相较基线有{degree}提升", "为应用落地提供依据",
            "对后续研究具有参考意义",
        ),
    },
    "life": {
        "subj": ("老师", "家长", "朋友", "同事", "学员", "孩子"),
        "verb": ("准备", "开始", "分享", "学习", "体验", "记录", "计划"),
        "obj": ("早餐", "运动", "读书", "旅行", "烹饪", "手作", "晨跑"),
        "tail": (
            "让一天更有节奏", "保持身心{degree}", "收获细小而确定的快乐", "在{place}形成习惯",
            "并与朋友{freq}交流",
        ),
    },
}


_TEMPLATES: Tuple[str, ...] = (
    "{time}，{subj}{modal}{verb}{obj}，{link}{tail}。",
    "在{place}，{subj}{modal}{verb}{obj}，{tail}。",
    "{subj}{freq}{verb}{obj}，{link}{tail}。",
    "若进展顺利，{subj}将{verb}{obj}，{tail}。",
    "经过一段时间，{subj}{verb}{obj}，{tail}。",
)


def _choice(rng: random.Random, items: Sequence[str]) -> str:
    return items[rng.randrange(len(items))]


def _compose_sentence(rng: random.Random, topic: str) -> str:
    topic = topic if topic in _TOPICS else "news"
    pools = _TOPICS[topic]
    template = _choice(rng, _TEMPLATES)
    # Slot sampling with light perturbations from _COMMON
    data = {
        "time": _choice(rng, _COMMON["time"]),
        "place": _choice(rng, _COMMON["place"]),
        "modal": _choice(rng, _COMMON["modal"]),
        "freq": _choice(rng, _COMMON["freq"]),
        "degree": _choice(rng, _COMMON["degree"]),
        "link": _choice(rng, _COMMON["link"]),
        "aux": _choice(rng, _COMMON["aux"]),
        "subj": _choice(rng, pools["subj"]),
        "verb": _choice(rng, pools["verb"]),
        "obj": _choice(rng, pools["obj"]),
        "tail": _choice(rng, pools["tail"]),
    }
    # Optional micro-perturbations: add an aux adverb before verb or obj.
    if rng.random() < 0.35:
        data["verb"] = data["aux"] + data["verb"]
    if rng.random() < 0.25:
        data["obj"] = data["obj"] + "计划"
    # Render and tidy spacing (Chinese punctuation only)
    sentence = template.format(**data)
    # Ensure ending punctuation
    if not sentence.endswith(("。", "！", "？")):
        sentence += "。"
    return sentence


def generate_sentences(count: int, topic: str = "news", seed: int | None = None) -> List[str]:
    """Generate a list of short, readable Chinese sentences.

    - count: number of sentences to generate (>=1)
    - topic: one of "news", "science", "life" (default: news)
    - seed: optional RNG seed for reproducibility
    """
    rng = random.Random(seed)
    n = max(1, int(count))
    return [_compose_sentence(rng, topic) for _ in range(n)]


def _topic_label(name: str) -> str:
    return "news" if name not in _TOPICS else name


def write_corpus(
    output: str | Path = "data",
    *,
    topics: Sequence[str] | None = None,
    lines_per_file: int | None = None,
    seed: int | None = None,
    # Compatibility (single-file mode)
    lines: int | None = None,
    topic_hint: str | None = None,
) -> List[Path]:
    """Write corpus files with generated sentences.

    Modes
    - Directory mode (default): output is a directory; writes seed_{topic}.txt per topic.
      - args: topics (default [news,science,life]), lines_per_file (default 800)
    - Single-file mode: output is a .txt path; writes exactly one file for topic_hint
      - args: lines (default 800), topic_hint required

    Returns a list of written file paths.
    """
    out = Path(output)
    base_seed = seed if seed is not None else random.randrange(1 << 30)

    # Single-file mode
    if out.suffix.lower() == ".txt":
        label = _topic_label(topic_hint or "news")
        n_lines = int(lines if lines is not None else (lines_per_file or 800))
        out.parent.mkdir(parents=True, exist_ok=True)
        lines_data = generate_sentences(max(1, n_lines), label, seed=base_seed)
        with out.open("w", encoding="utf-8") as f:
            for line in lines_data:
                f.write(line)
                f.write("\n")
        return [out]

    # Directory mode
    out.mkdir(parents=True, exist_ok=True)
    topics = list(topics) if topics is not None else ["news", "science", "life"]
    n_per = int(lines_per_file or lines or 800)
    written: List[Path] = []
    for i, topic in enumerate(topics):
        label = _topic_label(topic)
        file_path = out / f"seed_{label}.txt"
        local_seed = (base_seed + i * 1315423911) & 0x7FFFFFFF
        lines_data = generate_sentences(max(1, n_per), label, seed=local_seed)
        with file_path.open("w", encoding="utf-8") as f:
            for line in lines_data:
                f.write(line)
                f.write("\n")
        written.append(file_path)
    return written


def random_topics(seed: int | None = None) -> List[str]:
    """Return a randomized ordering of available topic names."""
    keys = list(_TOPICS.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys


if __name__ == "__main__":
    # Convenience: generate default 3 seed files.
    paths = write_corpus()
    print("Written:")
    for p in paths:
        print(p)
