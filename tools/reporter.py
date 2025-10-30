"""Human-readable reporting utilities for SMSA experiments (standard library only)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence
from tools.logger import CsvLogger
from tools.config import find_train_corpus_from_config
import statistics


def _format_meta(action: str | None, delta: float | None, reverted: bool | None) -> str:
    if not action:
        return "否"
    verb = "是"
    delta_str = f"{delta:+.3f}" if delta is not None else "0.000"
    flag = "已回滚" if reverted else "已生效"
    return f"{verb}（{action}, Δ={delta_str}, {flag}）"


def _format_metrics(data: Mapping[str, float]) -> str:
    parts: List[str] = []
    if "avg_return" in data:
        parts.append(f"平均回报 {data['avg_return']:.3f}")
    if "success_rate" in data:
        parts.append(f"成功率 {data['success_rate']:.2f}")
    if "ppl" in data:
        parts.append(f"困惑度 {data['ppl']:.2f}")
    if "energy" in data:
        parts.append(f"能耗 {data['energy']:.2f}")
    if not parts:
        parts.append("未提供核心指标")
    return "，".join(parts)


def write_episode_report(path: str | Path, data: Mapping[str, object]) -> None:
    """Append a markdown snippet summarising one episode."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    task = data.get("task", "未知任务")
    episode = data.get("episode", "?")
    cause = data.get("cause_prob_self", None)
    meta_text = _format_meta(
        str(data.get("meta_action") or "") or None,
        data.get("delta"),
        data.get("reverted"),
    )
    indicators = _format_metrics(
        {
            key: value
            for key, value in data.items()
            if isinstance(value, (int, float))
            and key
            in {"avg_return", "success_rate", "ppl", "energy", "nll", "loss"}
        }
    )
    plan = str(data.get("next_plan", "未指定后续计划"))
    cause_text = (
        f"Self-Model 预测自因概率 {cause:.2f}"
        if isinstance(cause, (int, float))
        else "Self-Model 自因概率未知"
    )
    calib_note = str(data.get("calibration_note") or "").strip()
    # 若未提供校准说明，尝试自动生成：读取同目录下 self_model_metrics.csv
    if not calib_note:
        try:
            metrics_csv = file_path.parent / "self_model_metrics.csv"
            rho, brier, win = compute_calibration_from_metrics(metrics_csv, window=50)
            rows = _read_csv_rows(metrics_csv)
            conf = 0.0
            acc = 0.0
            if rows:
                last = rows[-1]
                conf = float(last.get("conf_next") or 0.0)
                acc = float(last.get("acc_next") or 0.0)
            # 文案映射（按 conf 阈值）：
            # conf < 0.25 → “偏低”；0.25 ≤ conf ≤ 0.75 → “基本匹配”；> 0.75 → “偏高”
            if conf < 0.25:
                verdict = "偏低"
            elif conf <= 0.75:
                verdict = "基本匹配"
            else:
                verdict = "偏高"
            calib_note = (
                f"我对下一观测的置信度为 {conf:.2f}；过去 {win} 回合的校准相关 ρ={rho:.2f}，"
                f"说明我的置信{verdict}。"
            )
        except Exception:
            calib_note = ""
    domain_info = str(data.get("domains_summary") or "").strip()
    # 语料来源（优先来自调用方，其次来自全局配置）
    corpus_path = str(data.get("corpus_path") or "").strip()
    if not corpus_path:
        cfg_corpus = find_train_corpus_from_config()
        if cfg_corpus:
            corpus_path = str(cfg_corpus)
    delta_ppl = data.get("delta_ppl")
    extra_line = None
    if domain_info or (task == "lm" and isinstance(delta_ppl, (int, float))):
        parts: List[str] = []
        if domain_info:
            parts.append(f"语料学习：{domain_info}")
        if isinstance(delta_ppl, (int, float)):
            parts.append(f"验证困惑度改善 Δ{delta_ppl:+.3f}")
        extra_line = "- " + "，".join(parts)
    # RL 能耗箭头：与上一回合对比（基于 runs/daemon.csv 最近两条 RL 记录）
    energy_arrow = ""
    try:
        if str(task) == "rl":
            daemon_csv = file_path.parent / "daemon.csv"
            prev_e, curr_e = _last_two_task_energy(daemon_csv, task="rl")
            if prev_e is None or curr_e is None:
                energy_arrow = "→"
            else:
                if curr_e > prev_e + 1e-9:
                    energy_arrow = "↑"
                elif curr_e < prev_e - 1e-9:
                    energy_arrow = "↓"
                else:
                    energy_arrow = "→"
    except Exception:
        energy_arrow = ""

    arrow_suffix = (f"（能耗 vs 上一回合：{energy_arrow}）" if energy_arrow else "")
    lines = [
        f"### Episode {episode} · 任务：{task}",
        f"- 核心指标：{indicators}{arrow_suffix}",
    ]
    if extra_line:
        lines.append(extra_line)
    if corpus_path:
        lines.append(f"- 语料来源：{corpus_path}")
    # Post 空样本标注（tokens==0 或 spikes==0）：从 feed 元数据读取 tokens 与解码参数
    try:
        if str(task) == "post":
            energy_val = 0.0
            try:
                val = data.get("energy")
                energy_val = float(val) if isinstance(val, (int, float)) else 0.0
            except Exception:
                energy_val = 0.0
            feed_path = Path(str(data.get("text_path", "") or ""))
            feed_info = _read_feed_info(feed_path) if feed_path.is_file() else {}
            tokens_count = int(feed_info.get("tokens", -1)) if feed_info.get("tokens") is not None else -1
            is_empty = (tokens_count == 0) or (energy_val == 0.0)
            if is_empty:
                parts: List[str] = []
                if feed_info.get("T") is not None:
                    parts.append(f"T={feed_info['T']:.2f}")
                if feed_info.get("top_k") is not None:
                    parts.append(f"top_k={int(feed_info['top_k'])}")
                if feed_info.get("repeat_penalty") is not None:
                    parts.append(f"repeat_penalty={float(feed_info['repeat_penalty']):.2f}")
                if feed_info.get("trigram") is not None:
                    parts.append(f"trigram={int(feed_info['trigram'])}")
                param_text = " ".join(parts) if parts else "无解码参数记录"
                lines.append(f"- [空样本] 解码参数：{param_text}")
    except Exception:
        pass

    lines.extend(
        [
            f"- 本次自改：{meta_text}",
            f"- {cause_text}",
            (f"- {calib_note}" if calib_note else ""),
            f"- 后续计划：{plan}",
            "",
        ]
    )
    content = "\n".join(lines)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def _last_two_task_energy(csv_path: Path, task: str) -> tuple[float | None, float | None]:
    rows = _read_csv_rows(csv_path)
    rl_rows = [r for r in rows if r.get("task") == task]
    if not rl_rows:
        return None, None
    # 取最后两条 RL 记录，注意当前调用发生在日志追加之后
    if len(rl_rows) == 1:
        try:
            curr = float(rl_rows[-1].get("spikes") or 0.0)
        except (TypeError, ValueError):
            curr = None  # type: ignore[assignment]
        return None, curr
    try:
        prev = float(rl_rows[-2].get("spikes") or 0.0)
    except (TypeError, ValueError):
        prev = None  # type: ignore[assignment]
    try:
        curr = float(rl_rows[-1].get("spikes") or 0.0)
    except (TypeError, ValueError):
        curr = None  # type: ignore[assignment]
    return prev, curr


def _read_feed_info(feed_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        with feed_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("- Generated Tokens:"):
                    try:
                        info["tokens"] = int(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("- Temperature:"):
                    try:
                        info["T"] = float(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                elif line.startswith("- Attempts:"):
                    # 读取第一条尝试参数（若存在）
                    # 下一行通常以 "- [try#...] | T=... | top_k=... | repeat_penalty=... | trigram=..." 形式
                    continue  # header 本行略过，后续逐行处理
                elif line.startswith("- ") and "[try#" in line:
                    parts = [seg.strip() for seg in line[1:].split("|")]
                    for seg in parts:
                        if seg.startswith("top_k="):
                            try:
                                info["top_k"] = int(seg.split("=", 1)[1])
                            except Exception:
                                pass
                        elif seg.startswith("repeat_penalty="):
                            try:
                                info["repeat_penalty"] = float(seg.split("=", 1)[1])
                            except Exception:
                                pass
                        elif seg.startswith("trigram="):
                            try:
                                info["trigram"] = int(seg.split("=", 1)[1])
                            except Exception:
                                pass
                    # 只读取第一条尝试
                    break
    except OSError:
        return info
    return info


def _read_conf_acc(path: Path, window: int) -> tuple[list[float], list[float]]:
    vals_conf: List[float] = []
    vals_acc: List[float] = []
    if not path.exists():
        return vals_conf, vals_acc
    rows = _read_csv_rows(path)
    for row in rows[-window:]:
        try:
            conf = float(row.get("conf_next") or 0.0)
            acc = float(row.get("acc_next") or 0.0)
        except (TypeError, ValueError):
            continue
        vals_conf.append(conf)
        vals_acc.append(acc)
    return vals_conf, vals_acc


def _rankdata(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    sorted_pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        total_rank = 0.0
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            total_rank += j + 1
            j += 1
        avg_rank = total_rank / (j - i)
        for k in range(i, j):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def compute_calibration_from_metrics(metrics_csv: str | Path, window: int = 50) -> tuple[float, float, int]:
    path = Path(metrics_csv)
    confs, accs = _read_conf_acc(path, window)
    n = min(len(confs), len(accs))
    if n == 0:
        return 0.0, 0.0, 0
    confs = confs[-n:]
    accs = accs[-n:]
    rx = _rankdata(confs)
    ry = _rankdata(accs)
    rho = _pearson(rx, ry)
    brier = sum((c - a) ** 2 for c, a in zip(confs, accs)) / float(n)
    return rho, brier, n


def append_calibration_row(csv_path: str | Path, episode: int, rho: float, brier: float, window: int) -> None:
    path = Path(csv_path)
    logger = CsvLogger(path, ["episode", "spearman_rho", "brier", "window"])  # reuse CsvLogger
    logger.log({
        "episode": episode,
        "spearman_rho": f"{rho:.6f}",
        "brier": f"{brier:.6f}",
        "window": window,
    })
    logger.close()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def summarize_trend(csv_path: str | Path) -> str:
    """Provide three lines of Chinese text summarizing recent metrics."""
    path = Path(csv_path)
    rows = _read_csv_rows(path)
    if not rows:
        return f"{path} 暂无数据，无法生成趋势分析。"
    last = rows[-1]
    window = rows[-10:] if len(rows) >= 10 else rows
    def extract_float(row: Mapping[str, str], key: str) -> float | None:
        value = row.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    def stats(key: str) -> tuple[float | None, float | None]:
        values = [val for val in (extract_float(row, key) for row in window) if val is not None]
        if not values:
            return None, None
        avg = sum(values) / float(len(values))
        trend = values[-1] - values[0] if len(values) > 1 else 0.0
        return avg, trend
    ret_avg, ret_trend = stats("return")
    sr_avg, sr_trend = stats("success_rate")
    energy_avg, energy_trend = stats("energy")
    line1 = (
        f"最近 {len(window)} 条记录平均回报 {ret_avg:.3f}"
        if ret_avg is not None
        else "回报字段缺失，无法统计趋势"
    )
    if ret_trend is not None:
        line1 += f"，较窗口起点{'提升' if ret_trend >= 0 else '下降'} {abs(ret_trend):.3f}"
    line2 = (
        f"成功率均值 {sr_avg:.2f}，变化幅度 {sr_trend:+.2f}"
        if sr_avg is not None
        else "成功率缺失，改看其他指标"
    )
    line3 = (
        f"能耗均值 {energy_avg:.2f}，趋势 {energy_trend:+.2f}"
        if energy_avg is not None
        else "能耗数据为空，建议检查日志"
    )
    return "\n".join([line1, line2, line3])
