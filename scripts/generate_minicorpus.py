"""Generate a synthetic minimal Chinese commonsense corpus."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class Record:
    text: str
    domain: str
    intent: str
    tone: str
    urgency: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "text": self.text,
            "domain": self.domain,
            "intent": self.intent,
            "tone": self.tone,
            "urgency": self.urgency,
        }


@dataclass
class FactCategory:
    domain: str
    intents: Sequence[str]
    tones: Sequence[str]
    urgencies: Sequence[str]
    subjects: Sequence[str]
    objects: Sequence[str]
    predicates: Sequence[str]
    details: Sequence[str]
    contexts: Sequence[str] | None = None
    relations: Sequence[str] | None = None


CATEGORIES: List[FactCategory] = [
    FactCategory(
        domain="生活",
        intents=["陈述", "分享", "科普"],
        tones=["中性", "积极"],
        urgencies=["低"],
        subjects=[
            "苹果",
            "香蕉",
            "胡萝卜",
            "面包",
            "蜂蜜",
            "豆腐",
            "牛奶",
            "绿茶",
            "米饭",
            "鸡蛋",
            "酸奶",
            "玉米",
            "橄榄油",
            "西兰花",
            "红薯",
            "芝士",
        ],
        objects=["水果", "蔬菜", "主食", "饮品", "食材"],
        predicates=["是一种", "属于", "被认为是"],
        contexts=[
            "家庭早餐",
            "校园餐厅",
            "均衡饮食计划",
            "户外野餐",
            "地中海菜谱",
        ],
        details=[
            "富含膳食纤维，在{context}常被推荐",
            "在{context}能提供稳定能量",
            "在{context}中经常与谷物搭配",
            "帮助补充维生素，是{context}常见的选择",
            "味道清爽，适合{context}",
            "被许多人视为{context}的健康象征",
        ],
    ),
    FactCategory(
        domain="学习",
        intents=["科普", "记录", "陈述"],
        tones=["中性"],
        urgencies=["低", "中"],
        subjects=[
            "水循环",
            "牛顿第二定律",
            "元素周期表",
            "地球自转",
            "DNA",
            "光合作用",
            "勾股定理",
            "量子力学",
            "人工智能",
            "化学键",
            "贝叶斯定理",
            "太阳能",
        ],
        objects=["自然现象", "科学知识", "学习主题", "研究领域", "基础概念"],
        predicates=["是一种", "属于", "被归为", "体现为"],
        contexts=[
            "中学课堂",
            "科学实验",
            "课后讨论",
            "考试复习",
            "研究报告",
        ],
        details=[
            "帮助解释{context}里的重点原理",
            "常被用来分析能量变化",
            "是理解{context}的重要基础",
            "在{context}中用于验证推理",
            "让学生更容易把握抽象规律",
            "在{context}中连接多个学科的知识点",
        ],
    ),
    FactCategory(
        domain="出行",
        intents=["陈述", "分享", "记录"],
        tones=["中性"],
        urgencies=["低"],
        subjects=[
            "长江",
            "黄河",
            "珠穆朗玛峰",
            "撒哈拉沙漠",
            "杭州西湖",
            "京沪高铁",
            "青藏铁路",
            "丝绸之路",
            "故宫",
            "敦煌莫高窟",
            "港珠澳大桥",
            "海南环岛高铁",
        ],
        objects=["地标", "自然景观", "人文景点", "交通线路"],
        predicates=["被视为", "是一处", "属于", "代表着"],
        contexts=[
            "跨省旅行路线",
            "世界遗产名单",
            "探索计划",
            "地理教材",
            "旅游攻略",
        ],
        relations=[
            "中国",
            "亚洲",
            "丝路沿线",
            "沿海地区",
            "高原地带",
        ],
        details=[
            "在{context}中具有象征意义",
            "吸引大量游客，也见证着{relation}的历史记忆",
            "是认识{relation}地理的关键节点",
            "向旅客展示{relation}的文化魅力",
            "贯穿{relation}的重要交通脉络",
        ],
    ),
    FactCategory(
        domain="职场",
        intents=["陈述", "总结", "分享"],
        tones=["中性", "积极"],
        urgencies=["低", "中"],
        subjects=[
            "项目计划",
            "会议纪要",
            "数据分析",
            "云计算平台",
            "团队协作",
            "远程办公",
            "绩效评估",
            "职业培训",
            "客户需求",
            "风险管理",
            "产品愿景",
            "知识库",
        ],
        objects=["工作工具", "管理流程", "职业技能", "运营手段"],
        predicates=["是一项", "属于", "被用作", "被视为"],
        contexts=[
            "日常办公",
            "产品迭代",
            "战略复盘",
            "季度目标",
            "数字化转型",
        ],
        details=[
            "帮助团队在{context}保持一致",
            "让沟通在{context}保持透明",
            "经常用于{context}的决策参考",
            "提升{context}的执行效率",
            "在{context}中追踪关键指标",
            "把经验沉淀为{context}可复用的方法",
        ],
    ),
]


def build_sentence(rng: random.Random) -> Record:
    category = rng.choice(CATEGORIES)
    subject = rng.choice(category.subjects)
    obj = rng.choice(category.objects)
    predicate = rng.choice(category.predicates)
    context = rng.choice(category.contexts) if category.contexts else ""
    relation = rng.choice(category.relations) if category.relations else ""
    detail_template = rng.choice(category.details)
    detail = detail_template.format(
        subject=subject,
        object=obj,
        context=context,
        relation=relation,
    )
    detail = detail.replace("，，", "，").replace("。。", "。")
    text = f"{subject}{predicate}{obj}，{detail}。"
    tone = rng.choice(category.tones)
    intent = rng.choice(category.intents)
    urgency = rng.choice(category.urgencies)
    return Record(
        text=text,
        domain=category.domain,
        intent=intent,
        tone=tone,
        urgency=urgency,
    )


def generate_corpus(
    *,
    count: int,
    seed: int,
    output: Path,
) -> None:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: List[Record] = [build_sentence(rng) for _ in range(count)]
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mini Chinese corpus.")
    parser.add_argument("--count", type=int, default=3200, help="Number of sentences.")
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/minicorpus.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_corpus(count=args.count, seed=args.seed, output=Path(args.output))


if __name__ == "__main__":
    main()
