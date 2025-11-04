"""夜跑巡检报表（仅标准库）。

汇总最近 --hours 小时内的关键指标并输出 runs/nightly_report.md：
- RL: return/success/energy 趋势
- LM: ppl/Δppl 趋势
- Explain: post overall 均值
- Autogrow 次数（基于日志中的 n_hidden/grow/prune 关键字）
- 补丁回滚率（autopatch.log 中含“回滚”占比）
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _since(rows: List[Dict[str, str]], hours: int) -> List[Dict[str, str]]:
    if not rows:
        return []
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    out: List[Dict[str, str]] = []
    for r in rows:
        ts = r.get("timestamp")
        try:
            t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue
        if t >= cutoff:
            out.append(r)
    return out


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="夜跑巡检报表")
    p.add_argument("--hours", type=int, default=4, help="回溯小时数")
    args = p.parse_args(argv)

    daemon = _since(_read_rows(Path("runs/daemon.csv")), args.hours)
    sched = _since(_read_rows(Path("runs/scheduler.csv")), args.hours)
    ap_text = Path("runs/autopatch.log").read_text("utf-8", errors="ignore") if Path("runs/autopatch.log").exists() else ""

    # RL 指标
    rl = [r for r in daemon if r.get("task") == "rl"]
    lm = [r for r in daemon if r.get("task") == "lm"]
    post = [r for r in daemon if r.get("task") == "post"]

    def _avg(rows: List[Dict[str, str]], key: str) -> float:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key) or 0.0))
            except Exception:
                pass
        return sum(vals) / float(len(vals) or 1)

    report = ["# 夜跑巡检报表", ""]
    report.append("## 核心指标")
    report.append(f"- RL 平均回报：{_avg(rl,'reward'):.3f}；成功率近似：{_avg(rl,'metric_b'):.3f}；能耗：{_avg(rl,'spikes'):.2f}")
    report.append(f"- LM 验证困惑度：{_avg(lm,'metric_b'):.2f}；Δppl：{_avg(lm,'delta'):.3f}")
    report.append(f"- Post overall 均值：{_avg(post,'reward'):.3f}")

    # Autogrow 次数（粗略统计 note 中 n_hidden/grow/prune）
    grow_events = [r for r in sched if 'grow' in (r.get('note','')) or 'prune' in (r.get('note','')) or 'n_hidden' in (r.get('note',''))]
    report.append("")
    report.append("## 结构与补丁")
    report.append(f"- AutoGrow 记录：{len(grow_events)} 条（最近 {args.hours} 小时）")
    # 补丁回滚率
    rollbacks = ap_text.count("回滚")
    accepts = ap_text.count("提升")
    total = max(1, rollbacks + accepts)
    report.append(f"- AutoPatch 回滚率：{rollbacks/total:.2%}（采样窗口内）")

    out = Path("runs/nightly_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[nightly] 报表生成：{out}")


if __name__ == "__main__":
    main()

