"""变更摘要生成器：汇总参数改动、代码补丁与结构变更至 runs/change_summary.md。

仅使用标准库；从以下来源提炼：
- runs/daemon.csv：note/meta_action 字段提取参数变化（eta_e/v_th/inner_steps 等）；
- runs/autopatch.log：提取补丁 id、Δscore、是否回滚；
- runs/scheduler.csv：统计 task=link 与 grow/prune 相关注记（若存在）。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DAEMON_CSV = Path("runs/daemon.csv")
SCHED_CSV = Path("runs/scheduler.csv")
AUTOPATCH_LOG = Path("runs/autopatch.log")
OUT_MD = Path("runs/change_summary.md")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _extract_params(note: str) -> List[str]:
    fields = []
    for key in ("eta_e", "v_th", "inner_steps", "temperature", "top_k", "repeat_penalty"):
        m = re.search(rf"{key}=([0-9]+\.?[0-9]*)", note)
        if m:
            fields.append(f"{key}→{m.group(1)}")
    return fields


def _extract_autopatches(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        if "apply patch" in line or "apply_patch" in line or "decode:" in line or "surrogate:" in line:
            lines.append(line.strip())
        if "ab_evaluate:" in line or "退化" in line or "提升" in line:
            lines.append(line.strip())
    return lines


def summarize_changes(window: int = 100) -> str:
    daemon_rows = _read_csv(DAEMON_CSV)
    sched_rows = _read_csv(SCHED_CSV)
    ap_text = AUTOPATCH_LOG.read_text("utf-8", errors="ignore") if AUTOPATCH_LOG.exists() else ""

    md: List[str] = ["# 变更摘要", ""]
    # 参数改动
    md.append("## 参数改动（最近窗口）")
    param_events: List[str] = []
    for row in daemon_rows[-window:]:
        note = str(row.get("note") or "")
        params = _extract_params(note)
        if params:
            param_events.append(f"- ep{row.get('iteration','?')}：" + ", ".join(params))
    if not param_events:
        md.append("- 无显式参数改动记录")
    else:
        md.extend(param_events)

    # 代码补丁
    md.append("")
    md.append("## 代码补丁（AutoPatch）")
    for line in _extract_autopatches(ap_text)[-window:]:
        md.append(f"- {line}")
    if AUTOPATCH_LOG.exists():
        pass
    elif not ap_text:
        md.append("- 未找到 autopatch 日志")

    # 结构变更（grow/prune 与 link）
    md.append("")
    md.append("## 结构变更（扩容/裁剪/联动）")
    struct_lines = []
    for row in sched_rows[-window:]:
        task = row.get("task")
        note = row.get("note", "")
        if task in ("link",) or ("grow" in note or "prune" in note or "n_hidden" in note):
            struct_lines.append(f"- step{row.get('step','?')}: task={task} note={note}")
    if struct_lines:
        md.extend(struct_lines)
    else:
        md.append("- 近期无结构性变更记录")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    return str(OUT_MD)


__all__ = ["summarize_changes"]

