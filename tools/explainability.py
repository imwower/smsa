"""解释性与质量评估工具（仅标准库）。

功能概述：
- readability_score(text, lang="zh") -> 0..1：
  - 字符覆盖率（中文 / 常见标点 / 空白）
  - 句长与方差（长度适中、节奏平稳得分更高）
  - 3-gram 重复率惩罚
  - 类型多样性（按字 / 简易“词”粗分）
  - 标点节奏（结句标点与占比）

- context_score(text, topic_hint) -> 0..1：
  - 关键字命中率 + 主题一致性（包含式；无 hint 返回 0.5）

- self_explain_score(text) -> 0..1：
  - 自指：我 / 自己 / 此刻
  - 因果连接词：因为 / 因此 / 所以 / 于是 / 原因 / 导致 / 从而 / 以便 / 假设 / 如果
  - 五类词根命中计数：观测 / 诊断 / 行动 / 结果 / 计划（命中 ≥3 类得分高）

- explainability_index(text, topic_hint=None) -> dict：
  返回 {"readability": r, "context": c, "self_explain": e, "overall": 0.5*r + 0.2*c + 0.3*e, "notes": [...]}。

- suggest_actions(ei) -> list[dict]：
  将失败模式映射为可执行策略（解码调参 / SNN 时序 / NTP 再训练 / AutoPatch / 回滚）。

仅使用标准库，便于嵌入离线/守护进程工作流。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# --- 基础工具 ---------------------------------------------------------------

_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CN_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"[。！？!?…]+")
_PUNCT_SET = set("，。！？：；、,.!?;:()（）“”‘’《》—-…")
_SPACE_SET = set(" \t\n\r")


def _is_covered_char(ch: str) -> bool:
    return bool(_CN_CHAR_RE.match(ch)) or ch in _PUNCT_SET or ch in _SPACE_SET


def _sentences(text: str) -> List[str]:
    parts = [seg.strip() for seg in _SENT_SPLIT_RE.split(text) if seg.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _trigrams(chars: Sequence[str]) -> List[Tuple[str, str, str]]:
    return [tuple(chars[i : i + 3]) for i in range(max(0, len(chars) - 2))]


def _tokenize(text: str) -> List[str]:
    # 简易多通道：中文连续片段 + 英文/数字片段 + 其他落回单字符
    tokens: List[str] = []
    idx = 0
    while idx < len(text):
        m_cn = _CN_TOKEN_RE.match(text, idx)
        if m_cn:
            tokens.append(m_cn.group(0))
            idx = m_cn.end()
            continue
        m_en = _EN_TOKEN_RE.match(text, idx)
        if m_en:
            tokens.append(m_en.group(0))
            idx = m_en.end()
            continue
        ch = text[idx]
        if ch not in _SPACE_SET:
            tokens.append(ch)
        idx += 1
    return tokens


def _basic_stats(text: str) -> Dict[str, float | List[int]]:
    """提取可复用的统计量（供可读性与失败模式判定）。"""
    total = len(text)
    if total == 0:
        return {
            "coverage": 0.0,
            "mean_len": 0.0,
            "var_len": 0.0,
            "tri_rep": 1.0,
            "char_div": 0.0,
            "token_div": 0.0,
            "punct_ratio": 0.0,
        }

    # 字符覆盖率
    covered = sum(1 for ch in text if _is_covered_char(ch))
    coverage = covered / float(total)

    # 句长与方差
    lens = [len(s) for s in _sentences(text)]
    mean_len = (sum(lens) / float(len(lens))) if lens else 0.0
    var_len = (sum((l - mean_len) ** 2 for l in lens) / float(len(lens))) if lens else 0.0

    # 3-gram 重复率
    core_chars = [ch for ch in text if ch not in _SPACE_SET]
    tris = _trigrams(core_chars)
    if not tris:
        tri_rep = 0.0
    else:
        cnt = Counter(tris)
        repeats = sum(max(0, c - 1) for c in cnt.values())
        tri_rep = repeats / float(len(tris))

    # 多样性（字符与简易“词”）
    unique_chars = len(set(ch for ch in text if ch not in _SPACE_SET))
    char_div = unique_chars / float(max(1, len(core_chars)))
    tokens = _tokenize(text)
    token_div = len(set(tokens)) / float(max(1, len(tokens)))

    # 标点节奏
    punct_count = sum(1 for ch in text if ch in _PUNCT_SET)
    punct_ratio = punct_count / float(total)

    return {
        "coverage": coverage,
        "mean_len": mean_len,
        "var_len": var_len,
        "tri_rep": tri_rep,
        "char_div": char_div,
        "token_div": token_div,
        "punct_ratio": punct_ratio,
    }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --- 评分函数 ----------------------------------------------------------------

def readability_score(text: str, lang: str = "zh") -> float:
    """返回 0~1 的可读性分数（越高越好）。

    设计目标：在中文短文/段落上，能区分“乱码 / 重复 / 句子碎裂”与“自然、清晰”的文本。
    """
    del lang  # 当前按中文优先的启发式
    stats = _basic_stats(text)
    coverage = float(stats["coverage"])  # type: ignore[index]
    mean_len = float(stats["mean_len"])  # type: ignore[index]
    var_len = float(stats["var_len"])  # type: ignore[index]
    tri_rep = float(stats["tri_rep"])  # type: ignore[index]
    char_div = float(stats["char_div"])  # type: ignore[index]
    token_div = float(stats["token_div"])  # type: ignore[index]
    punct_ratio = float(stats["punct_ratio"])  # type: ignore[index]

    # 句长得分：8~40 为理想区间
    if mean_len <= 0:
        s_len = 0.0
    elif mean_len < 8:
        s_len = mean_len / 8.0
    elif mean_len <= 40:
        s_len = 1.0
    else:
        s_len = _clamp(1.0 - (mean_len - 40.0) / 40.0)

    # 方差惩罚（方差/均值近似节奏性，越大越差）
    var_ratio = var_len / float(mean_len + 1e-6)
    s_var = _clamp(1.0 - var_ratio / 50.0)
    s_sentence = 0.7 * s_len + 0.3 * s_var

    # 重复惩罚（3-gram）
    s_repeat = _clamp(1.0 - tri_rep)

    # 多样性
    s_div = _clamp(0.5 * char_div + 0.5 * token_div)

    # 标点节奏（2% ~ 15% 视为合理）
    if punct_ratio <= 0:
        s_punct = 0.2
    elif punct_ratio < 0.02:
        s_punct = 0.4 + punct_ratio * 30.0
    elif punct_ratio <= 0.15:
        s_punct = 1.0
    else:
        s_punct = _clamp(1.0 - (punct_ratio - 0.15) * 3.0)

    # 覆盖率（过滤乱码）
    s_cover = coverage

    score = (
        0.25 * s_cover
        + 0.25 * s_sentence
        + 0.25 * s_repeat
        + 0.15 * s_div
        + 0.10 * s_punct
    )
    return _clamp(score)


def context_score(text: str, topic_hint: Optional[str] = None) -> float:
    """上下文相关性打分：无 hint 返回 0.5；有 hint 时按命中率映射到 0.3~1.0。"""
    if not topic_hint:
        return 0.5
    # 从 hint 抽取关键词（中文段 / 英文词）
    hint_tokens = set(_CN_TOKEN_RE.findall(topic_hint)) | set(_EN_TOKEN_RE.findall(topic_hint))
    if not hint_tokens:
        hint_tokens = {topic_hint.strip()}
    text_tokens = set(_CN_TOKEN_RE.findall(text)) | set(_EN_TOKEN_RE.findall(text))
    hits = sum(1 for tok in hint_tokens if tok and (tok in text_tokens or tok in text))
    hit_rate = hits / float(max(1, len(hint_tokens)))
    # 命中率 → 分数（保留一定底线）
    return _clamp(0.3 + 0.7 * hit_rate)


def self_explain_score(text: str) -> float:
    """“自指 + 因果 + 任务五要素”综合评分。"""
    self_refs = ["我", "自己", "此刻"]
    connectors = ["因为", "因此", "所以", "于是", "原因", "导致", "从而", "以便", "假设", "如果"]
    categories = {
        "观测": ["观察", "观测", "记录", "看到", "识别"],
        "诊断": ["诊断", "分析", "判断", "评估", "定位"],
        "行动": ["行动", "执行", "降低", "增加", "调整", "尝试", "修改", "训练"],
        "结果": ["结果", "回报", "成功", "失败", "改善", "下降", "提升", "曲线"],
        "计划": ["计划", "下一步", "目标", "准备", "打算", "随后", "将", "之后"],
    }

    s_self = 1.0 if any(tok in text for tok in self_refs) else 0.0
    conn_count = sum(text.count(tok) for tok in connectors)
    s_conn = _clamp(conn_count / 3.0)  # 3 次及以上视为充分

    cat_hits = 0
    for _, vocab in categories.items():
        if any(tok in text for tok in vocab):
            cat_hits += 1
    s_cats = _clamp(cat_hits / 5.0)

    # 综合：强调五要素与因果，其次自指
    score = 0.40 * s_cats + 0.35 * s_conn + 0.25 * s_self
    return _clamp(score)


# --- 解释性索引与建议 -------------------------------------------------------

def explainability_index(text: str, topic_hint: Optional[str] = None) -> Dict[str, object]:
    r = readability_score(text)
    c = context_score(text, topic_hint)
    e = self_explain_score(text)

    # 失败模式标签（用于建议）
    notes: List[str] = []
    stats = _basic_stats(text)
    rep = float(stats["tri_rep"])  # type: ignore[index]
    mean_len = float(stats["mean_len"])  # type: ignore[index]
    # 复制一份“连接词/自指”分析以标注缺失
    connectors = ["因为", "因此", "所以", "于是", "原因", "导致", "从而", "以便", "假设", "如果"]
    if rep > 0.2:
        notes.append("重复过多")
    if c < 0.4:
        notes.append("主题稀薄")
    if not any(tok in text for tok in ["我", "自己", "此刻"]):
        notes.append("缺少自指")
    if sum(text.count(tok) for tok in connectors) == 0:
        notes.append("缺少因果")
    if mean_len < 6:
        notes.append("短句碎裂")

    overall = _clamp(0.5 * r + 0.2 * c + 0.3 * e)
    return {
        "readability": r,
        "context": c,
        "self_explain": e,
        "overall": overall,
        "notes": notes,
    }


def suggest_actions(ei: Mapping[str, object]) -> List[Dict[str, object]]:
    """根据失败模式生成可执行策略（按优先级排序）。"""
    notes = set(str(x) for x in ei.get("notes", []) if x)
    overall = float(ei.get("overall", 0.0) or 0.0)
    actions: List[Dict[str, object]] = []

    def add(name: str, why: str, params: Mapping[str, object] | None = None, priority: int = 5) -> None:
        actions.append({"action": name, "why": why, "params": dict(params or {}), "priority": priority})

    # 1) 文本重复与碎裂 → 解码调参 + SNN 时序
    if "重复过多" in notes:
        add("decode:repeat_penalty↑", "3-gram 重复率较高，建议提高惩罚", {"repeat_penalty": 1.2}, 1)
        add("decode:top_k↑", "丰富候选以减少机械重复", {"top_k": ">=50"}, 2)
        add("decode:temperature↓", "减小采样温度抑制回圈", {"temperature": 0.7}, 3)
    if "短句碎裂" in notes:
        add("snn:inner_steps↑", "句子过短，增加积分步数以延长关联", {"inner_steps": "+4"}, 2)
        add("snn:v_th↓", "阈值略降以提升发放连续性", {"v_th": "-0.05"}, 3)
        add("snn:lam_e↑/eta_e↓", "增加记忆跨度，减小抖动", {"lam_e": "+0.05", "eta_e": "-20%"}, 4)

    # 2) 主题稀薄 → 采样新主题语料 + 温度微调
    if "主题稀薄" in notes:
        add("ntp:resample_domain", "主题命中低，建议采样 1–2k 行目标主题语料继续学", {"lines": 1500}, 1)
        add("decode:temperature↓", "降低温度以收敛到主题词", {"temperature": 0.85}, 3)

    # 3) 因果/自指缺失 → 模板引导 + 解码/时序
    if "缺少因果" in notes:
        add("prompt:inject_connectives", "引入‘因为…所以…/于是…’模板", {"connectives": ["因为", "所以", "于是"]}, 2)
        add("decode:top_k↓", "轻降 top_k 以形成连贯因果链", {"top_k": 30}, 3)
    if "缺少自指" in notes:
        add("prompt:first_person", "引导‘我/自己/此刻’的自指句式", {}, 2)

    # 4) 全局策略：AutoPatch 与回滚
    if overall < 0.45:
        add("autopatch:surrogate_expr", "尝试更窄/更平滑的替代导数以稳态输出", {"candidate": "surrogate_rect"}, 5)
        add("rollback", "若多轮无改善，回滚前次改动", {}, 6)

    # 去重并按优先级排序
    seen = set()
    uniq: List[Dict[str, object]] = []
    for act in sorted(actions, key=lambda x: x.get("priority", 5)):
        key = (act["action"], tuple(sorted(act.get("params", {}).items())))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(act)
    return uniq


__all__ = [
    "readability_score",
    "context_score",
    "self_explain_score",
    "explainability_index",
    "suggest_actions",
]


if __name__ == "__main__":
    # 最小示例（不会作为单测运行，仅用于手工验证）
    bad = "x9*&^$今...今今...因此因此因此。。。"
    good = (
        "我观察到今天的目标是完成训练。"
        "因为奖励曲线放缓，于是我降低温度并增加 inner_steps。"
    )
    ei_bad = explainability_index(bad, "训练")
    ei_good = explainability_index(good, "训练")
    print("bad:", ei_bad)
    print("good:", ei_good)
    print("suggest:", suggest_actions(ei_bad))

