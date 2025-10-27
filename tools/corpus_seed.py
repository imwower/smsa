"""种子语料生成器：结合模板与同义词，实现可控主题句子扩展。"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tools.readability import readability_score


__all__ = ["generate_sentences", "write_corpus", "random_topics"]


@dataclass(frozen=True)
class Template:
    subject: Tuple[str, ...]
    verb: Tuple[str, ...]
    object: Tuple[str, ...]
    embellish: Tuple[str, ...]


TOPIC_TEMPLATES: Dict[str, List[Template]] = {
    "科技": [
        Template(
            subject=("研究团队", "年轻工程师", "实验室"),  # 主语
            verb=("推出", "细心雕琢", "加速打造"),  # 动词
            object=("原型设备", "算法框架", "智能芯片"),  # 宾语
            embellish=("像晨星一样点亮了创业工坊", "用比喻的手法描绘未来的轮廓", "在夜班的霓虹中诉说着温度"),  # 修饰
        ),
        Template(
            subject=("数据科学家", "城市传感网", "云端服务器"),
            verb=("悄悄记录", "耐心蒸馏", "稳步守护"),
            object=("脉搏似的讯号", "动态指数", "使用者的信任"),
            embellish=("像诗人收集晨露", "让复杂规律化作可以握住的语言", "在雨夜里仍保持温柔光亮"),
        ),
    ],
    "教育": [
        Template(
            subject=("班主任", "青年导师", "图书馆的志愿者"),
            verb=("设计", "编织", "分享"),
            object=("探究课题", "跨学科活动", "一杯温热的故事"),
            embellish=("像春风拂面般唤醒好奇心", "让教室在傍晚泛起金色波纹", "引导孩子们把星光放进口袋"),
        ),
        Template(
            subject=("旧校舍", "新课程", "午后阳光"),
            verb=("陪伴", "滋养", "照亮"),
            object=("求知若渴的孩子", "编织梦想的老师", "翻飞的笔记"),
            embellish=("像老友般交换心事", "散发着粉笔与咖啡交织的香气", "在笑声里写下慢慢凝固的成长"),
        ),
    ],
    "旅行": [
        Template(
            subject=("背包客", "清晨的列车", "海边的灯塔"),
            verb=("描绘", "见证", "守望"),
            object=("蜿蜒的山路", "穿城而过的风", "漂浮的回忆"),
            embellish=("像老电影般泛着颗粒感", "在车窗上铺开一层柔光", "教人把疲惫折叠成浪花"),
        ),
        Template(
            subject=("旅伴", "云海", "古城石阶"),
            verb=("相互鼓励", "慢慢散开", "轻声诉说"),
            object=("未知的岔路", "日出的惊喜", "旧时故事"),
            embellish=("像夏夜烟火照亮彼此", "让心情像纸鹤般翩飞", "在行囊里塞满香味和笑语"),
        ),
    ],
    "健康": [
        Template(
            subject=("晨跑者", "营养师", "社区诊所"),
            verb=("编织", "记录", "守护"),
            object=("有序的生活节奏", "体检数据", "邻里的安稳呼吸"),
            embellish=("像小溪一样滋润心田", "让身体与思想保持弹性", "用温暖灯光驱散寒意"),
        ),
        Template(
            subject=("手作厨房", "瑜伽老师", "呼吸训练"),
            verb=("调和", "提醒", "接住"),
            object=("五色蔬果", "放慢的步伐", "深夜的疲惫"),
            embellish=("把平凡午后染成清新的绿", "像树影般引导人伸展", "在胸腔里拉开一段晴朗的空间"),
        ),
    ],
    "城市": [
        Template(
            subject=("街角咖啡馆", "地铁列车", "暮色里的天桥"),
            verb=("翻涌", "承载", "串联"),
            object=("碎片化的故事", "节奏分明的脚步", "夜行者的心事"),
            embellish=("像极光一样为霓虹涂上柔焦", "在拥挤却温柔的秩序中流淌", "把楼宇之间的距离织成旋律"),
        ),
        Template(
            subject=("露天市集", "晨雾", "高楼玻璃"),
            verb=("散播", "轻拥", "折射"),
            object=("烟火气息", "清新的凉意", "百年老街的记忆"),
            embellish=("像山风吹动风铃般叮咚作响", "在环城公路上写下一句诗", "让行人脸庞染上一层金色的光晕"),
        ),
    ],
}


BASE_TEMPLATES: List[Template] = [
    Template(
        subject=("清晨", "夜色", "新芽"),
        verb=("唤醒", "照亮", "守候"),
        object=("城市的节奏", "远方的希望", "心底的秘密"),
        embellish=("像老友般拍肩", "带着月光的温度", "让繁忙的人也放慢脚步"),
    )
]


def generate_sentences(n: int, topic_hint: str | None = None) -> List[str]:
    """根据主题提示生成 n 条温和的中文句子。"""
    if n <= 0:
        return []
    topic = _normalize_topic(topic_hint)
    templates = list(TOPIC_TEMPLATES.get(topic, [])) + BASE_TEMPLATES
    rng = random.Random()
    sentences: List[str] = []
    seen: set[str] = set()
    while len(sentences) < n:
        tpl = rng.choice(templates)
        subject = _pick_variant(rng, tpl.subject)
        verb = _pick_variant(rng, tpl.verb)
        obj = _pick_variant(rng, tpl.object)
        embellish = _pick_variant(rng, tpl.embellish)
        clause = f"{subject}{rng.choice(('正', '正在', '悄悄', '缓缓', '轻轻'))}{verb}{obj}"
        sentence = f"{clause}，{embellish}。"
        sentence = _add_variation(sentence, rng)
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    return sentences


def write_corpus(path: str, lines: int, topic_hint: str | None = None) -> None:
    """将生成的句子追加写入语料文件，避免重复并确保整体可读性。"""
    if lines <= 0:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(path_obj)
    needed = max(0, lines - len(existing))
    if needed <= 0:
        return
    buffer: List[str] = []
    attempts = 0
    while len(buffer) < needed and attempts < needed * 4:
        attempts += 1
        batch = generate_sentences(4, topic_hint)
        for sentence in batch:
            if sentence not in existing and sentence not in buffer:
                buffer.append(sentence)
                if len(buffer) >= needed:
                    break

    if not buffer:
        return
    content = "\n".join(buffer) + "\n"
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(content)

    _record_readability(path_obj, buffer, topic_hint)


def random_topics() -> List[str]:
    """返回一组常用主题，供批量生成或 UI 下拉选项使用。"""
    topics = list(TOPIC_TEMPLATES.keys())
    random.shuffle(topics)
    return topics


def _normalize_topic(topic_hint: str | None) -> str:
    if not topic_hint:
        return "通用"
    hint = topic_hint.strip()
    if not hint:
        return "通用"
    for topic in TOPIC_TEMPLATES:
        if topic in hint:
            return topic
    return "通用"


def _pick_variant(rng: random.Random, options: Sequence[str]) -> str:
    if not options:
        return ""
    choice = rng.choice(options)
    if rng.random() < 0.15:
        synonyms = {
            "守护": ("守候", "呵护", "护航"),
            "照亮": ("点亮", "烘托", "映照"),
            "设计": ("策划", "盘活", "雕刻"),
            "分享": ("传递", "递送", "安放"),
            "守望": ("眺望", "遥望", "守候"),
            "描绘": ("勾勒", "描摹", "书写"),
            "旅伴": ("同行者", "驴友", "伙伴"),
        }
        replacement = synonyms.get(choice)
        if replacement:
            return rng.choice(replacement)
    return choice


def _add_variation(sentence: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        sentence = sentence.replace("，", "，仿佛", 1)
    if rng.random() < 0.25:
        sentence = sentence.replace("。", "，让人安心。", 1)
    return sentence


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _record_readability(path: Path, lines: Sequence[str], topic: str | None) -> None:
    if not lines:
        return
    scores = [readability_score(line) for line in lines]
    avg_score = sum(scores) / len(scores)
    report_path = path.with_suffix(path.suffix + ".meta")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{len(lines)} lines appended | avg_readability={avg_score:.4f} | topic={topic or ''}\n"
        )


if __name__ == "__main__":
    # 简易手动检查，以科技主题生成 5 句并写入示例文件。
    demo_lines = generate_sentences(5, topic_hint="科技升级")
    for line in demo_lines:
        print(line)
    write_corpus("data/demo_seed.txt", lines=20, topic_hint="科技")
