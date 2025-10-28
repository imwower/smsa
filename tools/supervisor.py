"""自适应重试监督器（仅标准库）。

SupervisedPost：
- 首次生成（spike_writer）并打分（explainability_index）
- 若整体/自解释分数低于阈值，调用 suggest_actions 生成改进动作序列
- 依次尝试：
  1) 解码器参数（温度 / top‑k / 重复惩罚）
  2) SNN 运行参数（若暴露；当前跳过并记录）
  3) 继续训练（scripts.snn_text_lm.train_lines）
  4) AutoPatch（tools.contracts.enforce_code_patch，触发安全评估与回滚）
- 每次重试后重新评估，保留最优结果
- 将每轮 Observation→Diagnosis→Intervention→Outcome→Next Step 记录到 runs/explain_log.md

仅使用标准库依赖。
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _ts() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")


@dataclass
class DecodeConfig:
    temperature: float = 1.0
    top_k: int = 50
    repeat_penalty: float = 1.1


@dataclass
class AttemptRecord:
    idx: int
    decode: DecodeConfig
    actions: List[str] = field(default_factory=list)
    text: str = ""
    score_overall: float = 0.0
    score_readability: float = 0.0
    score_context: float = 0.0
    score_self: float = 0.0
    notes: List[str] = field(default_factory=list)


class SupervisedPost:
    def __init__(
        self,
        *,
        attempts: int = 3,
        thresholds: Optional[Mapping[str, float]] = None,
        log_path: str | Path = "runs/explain_log.md",
    ) -> None:
        self.attempts = max(1, attempts)
        base = {"overall": 0.62, "self_explain": 0.40}
        if thresholds:
            base.update({k: float(v) for k, v in thresholds.items()})
        self.thresholds = base
        self.log_path = Path(log_path)

    # --- 执行动作 -----------------------------------------------------------
    @staticmethod
    def _apply_decode_action(cfg: DecodeConfig, action: str) -> Optional[str]:
        if action.startswith("decode:temperature"):
            if action.endswith("↓"):
                old = cfg.temperature
                cfg.temperature = max(0.6, cfg.temperature * 0.9)
                return f"temperature {old:.2f}→{cfg.temperature:.2f}"
            if action.endswith("↑"):
                old = cfg.temperature
                cfg.temperature = min(1.5, cfg.temperature * 1.1)
                return f"temperature {old:.2f}→{cfg.temperature:.2f}"
        if action.startswith("decode:top_k"):
            if action.endswith("↑"):
                old = cfg.top_k
                cfg.top_k = min(120, int(round(cfg.top_k * 1.5)))
                return f"top_k {old}→{cfg.top_k}"
            if action.endswith("↓"):
                old = cfg.top_k
                cfg.top_k = max(10, int(round(cfg.top_k * 0.6)))
                return f"top_k {old}→{cfg.top_k}"
        if action.startswith("decode:repeat_penalty"):
            if action.endswith("↑"):
                old = cfg.repeat_penalty
                cfg.repeat_penalty = min(2.0, cfg.repeat_penalty + 0.1)
                return f"repeat_penalty {old:.2f}→{cfg.repeat_penalty:.2f}"
        return None

    @staticmethod
    def _apply_ntp_action(action: str) -> Optional[str]:
        if not action.startswith("ntp:resample_domain"):
            return None
        lines = 1200
        # 从动作 params 提示中提取行数（若存在）
        # 这里无法解析结构化参数，只做固定小步训练
        try:
            from scripts.snn_text_lm import train_lines  # 延迟导入
        except Exception:
            return "ntp:resample_domain skipped (import error)"
        stats = train_lines(num_lines=lines, seed=int(time.time()) & 0xFFFF)
        return f"ntp:resample_domain lines={lines} ppl={stats.get('ppl')}"

    @staticmethod
    def _apply_autopatch_action(action: str) -> Optional[str]:
        if not action.startswith("autopatch:"):
            return None
        try:
            from tools import contracts as qc
        except Exception:
            return "autopatch skipped (contracts import error)"
        candidate = "surrogate_rect"
        try:
            decision = qc.enforce_code_patch(candidate)
            if decision.accepted:
                return f"autopatch accepted Δ={decision.delta_score:+.3f} net={decision.net_benefit:+.3f}"
            return f"autopatch veto ({decision.reason})"
        except Exception as exc:  # 防御式
            return f"autopatch error: {exc}"

    # --- 生成与打分 ---------------------------------------------------------
    @staticmethod
    def _generate(topic: Optional[str], max_len: int, cfg: DecodeConfig) -> Tuple[str, Mapping[str, float], List[str]]:
        # 动态注入解码参数到 spike_writer 模块的全局（标准库能力）
        from scripts import spike_writer as sw
        from tools.explainability import explainability_index

        sw.DECODE_TOP_K = int(cfg.top_k)
        sw.DECODE_REPEAT_PENALTY = float(cfg.repeat_penalty)
        result = sw.spike_generate(
            max_len=max_len,
            seed_text="",
            temperature=float(cfg.temperature),
            topic_hint=topic,
        )
        ei = explainability_index(result.text, topic)
        notes = list(ei.get("notes", []))
        score = {
            "overall": float(ei.get("overall", 0.0) or 0.0),
            "readability": float(ei.get("readability", 0.0) or 0.0),
            "context": float(ei.get("context", 0.0) or 0.0),
            "self_explain": float(ei.get("self_explain", 0.0) or 0.0),
        }
        return result.text, score, notes

    # --- 主流程 -------------------------------------------------------------
    def run(self, *, topic: str | None = None, max_len: int = 160) -> Dict[str, object]:
        from tools.explainability import explainability_index, suggest_actions

        log_file = self.log_path
        best: Optional[AttemptRecord] = None
        history: List[AttemptRecord] = []
        cfg = DecodeConfig()
        executed_actions: List[str] = []

        for idx in range(1, self.attempts + 1):
            text, score, notes = self._generate(topic, max_len, cfg)
            rec = AttemptRecord(
                idx=idx,
                decode=DecodeConfig(cfg.temperature, cfg.top_k, cfg.repeat_penalty),
                actions=[],
                text=text,
                score_overall=score["overall"],
                score_readability=score["readability"],
                score_context=score["context"],
                score_self=score["self_explain"],
                notes=notes,
            )
            history.append(rec)

            # 日志：Observation / Diagnosis
            obs = [
                f"## Attempt {idx} · {_ts()}",
                f"Observation: overall={rec.score_overall:.4f} self={rec.score_self:.4f} topic={topic or ''}",
                f"Diagnosis: notes={', '.join(rec.notes) if rec.notes else '无'}",
            ]

            # 更新最优
            if best is None or rec.score_overall > best.score_overall:
                best = rec

            # 阈值满足或已到最后一轮 → 记录 Outcome 并退出
            if (
                rec.score_overall >= self.thresholds["overall"]
                and rec.score_self >= self.thresholds["self_explain"]
            ) or idx == self.attempts:
                _append_log(log_file, obs + [f"Outcome: accept (stop)", "Next Step: stop", ""])
                break

            # 生成建议
            ei = explainability_index(text, topic)
            actions = suggest_actions(ei)
            # 限制每轮最多执行 3 个动作（按优先级）
            actions = actions[:3]
            applied: List[str] = []

            # 依次尝试：解码器 / SNN / NTP / AutoPatch
            for act in actions:
                name = str(act.get("action", ""))
                if name.startswith("decode:"):
                    desc = self._apply_decode_action(cfg, name)
                    if desc:
                        applied.append(desc)
                        continue
                if name.startswith("snn:"):
                    applied.append("snn params skipped (not exposed)")
                    continue
                if name.startswith("ntp:"):
                    desc = self._apply_ntp_action(name)
                    if desc:
                        applied.append(desc)
                        continue
                if name.startswith("autopatch:"):
                    desc = self._apply_autopatch_action(name)
                    if desc:
                        applied.append(desc)
                        continue

            rec.actions.extend(applied)
            executed_actions.extend(applied)

            # Outcome + Next Step
            next_hint = actions[0]["action"] if actions else "none"
            _append_log(
                log_file,
                obs
                + [
                    f"Intervention: {', '.join(applied) if applied else 'none'}",
                    f"Outcome: will retry (next: {next_hint})",
                    "Next Step: retry",
                    "",
                ],
            )
            # 下一轮将以新 cfg 再次生成

        # 返回最优结果
        result = {
            "text": best.text if best else "",
            "score": best.score_overall if best else 0.0,
            "details": {
                "readability": best.score_readability if best else 0.0,
                "context": best.score_context if best else 0.0,
                "self_explain": best.score_self if best else 0.0,
                "notes": best.notes if best else [],
            },
            "attempts": len(history),
            "best_action": (history[-1].actions[0] if history and history[-1].actions else "baseline"),
            "actions": executed_actions,
            "log_path": str(self.log_path),
        }
        return result


__all__ = ["SupervisedPost"]

