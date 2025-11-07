"""Background daemon that cycles through RL/LM/Meta/Post tasks."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta.autoadapt import MetaLearner
from scripts.snn_text_lm import train_lines
from scripts.spike_writer import _write_feed, spike_generate, GenerationResult
from tools.supervisor import SupervisedPost
from scripts.train_gridworld import train_once
from tools.corpus import DomainSampler
from tools.config import write_corpus_config_for_path, find_train_corpus_from_config
import json
from tools.reporter import write_episode_report
from tools.scheduler import Scheduler


DAEMON_CSV_PATH = Path("runs/daemon.csv")
DAEMON_FIELDS = [
    "timestamp",
    "iteration",
    "task",
    "reward",
    "energy_penalty",
    "metric_a",
    "metric_b",
    "spikes",
    "meta_action",
    "delta",
    "reverted",
    "relearned",
    "retuned",
    "patched",
    "grown",
    "cooldown",
    "note",
]

LM_STATE_PATH = Path("runs/corpus_state.json")
ENERGY_COEFF_RL = 0.001
ENERGY_COEFF_LM = 0.0005
ENERGY_COEFF_POST = 0.0002
# 连续不达标次数阈值（默认 2，允许通过 --post-fail-max 调整至 2~3）
POST_FAIL_MAX = 2

_RL_HISTORY: Dict[str, float | None] = {"avg_return": None}


def ensure_daemon_csv() -> None:
    DAEMON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DAEMON_CSV_PATH.exists():
        with DAEMON_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DAEMON_FIELDS)
            writer.writeheader()


def append_daemon_row(row: Dict[str, object]) -> None:
    ensure_daemon_csv()
    payload = {key: row.get(key, "") for key in DAEMON_FIELDS}
    with DAEMON_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAEMON_FIELDS)
        writer.writerow(payload)


def build_domain_sampler(corpus_path: str | None = None) -> DomainSampler | None:
    """Instantiate a DomainSampler and register corpus files from config or explicit path."""
    LM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        sampler = DomainSampler(state_json=LM_STATE_PATH, split="train", window=10, ucb_c=0.5)
    except Exception as exc:
        print(f"[lm] 无法构建 DomainSampler: {exc}")
        return None

    # 1) 显式参数优先
    if corpus_path:
        try:
            sampler.register(corpus_path, split="train")
        except Exception as exc:
            print(f"[lm] 注册语料失败: {corpus_path} err={exc}")
        return sampler

    # 2) 读取全局配置
    train_path = find_train_corpus_from_config()
    if train_path:
        try:
            sampler.register(train_path, split="train")
        except Exception:
            pass
        return sampler

    # 3) 回退（优先 HF 展开）：data/hf/**/train.txt → data/*.txt
    hf_candidates = sorted(Path("data/hf").glob("**/train.txt"))
    if hf_candidates:
        try:
            sampler.register(str(hf_candidates[0]), split="train")
        except Exception:
            pass
        return sampler

    fallback = list(Path("data").glob("*.txt"))
    for p in fallback:
        try:
            sampler.register(p, split="train")
        except Exception:
            continue
    return sampler


def read_last_iteration(path: Path = DAEMON_CSV_PATH) -> int:
    """Load the last recorded iteration so numbering stays monotonic."""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            last_iteration = 0
            for row in reader:
                raw_value = (row.get("iteration") or "").strip()
                if not raw_value or raw_value == "iteration":
                    continue
                try:
                    last_iteration = int(raw_value)
                except ValueError:
                    continue
            return last_iteration
    except FileNotFoundError:
        return 0


class SimpleAutoAgent:
    def __init__(self) -> None:
        self.eta = 0.05
        self.vth = 0.6

    def apply_modification(self, action: str) -> Tuple[bool, str]:
        info = ""
        if action == "eta_up":
            self.eta = min(self.eta * 1.2, 0.3)
            info = f"eta={self.eta:.3f}"
            return True, info
        if action == "eta_down":
            self.eta = max(self.eta * 0.8, 0.005)
            info = f"eta={self.eta:.3f}"
            return True, info
        if action == "vth_up":
            self.vth = min(self.vth + 0.05, 1.2)
            info = f"v_th={self.vth:.3f}"
            return True, info
        if action == "vth_down":
            self.vth = max(self.vth - 0.05, 0.2)
            info = f"v_th={self.vth:.3f}"
            return True, info
        return False, "unsupported"


def _parse_log_delta(message: str) -> Tuple[str, float, bool]:
    action = "none"
    delta = 0.0
    reverted = "reverted" in message
    if "action=" in message:
        action = message.split("action=")[1].split()[0]
    if "delta=" in message:
        try:
            delta_str = message.split("delta=")[1].split()[0]
            delta = float(delta_str)
        except (IndexError, ValueError):
            delta = 0.0
    return action, delta, reverted


class AutoAdaptLoop:
    def __init__(self) -> None:
        self.meta = MetaLearner(
            window=4,
            min_delta=0.01,
            ab_episodes=5,
            actions=[
                "eta_up",
                "eta_down",
                "vth_up",
                "vth_down",
                # 注入代码补丁候选（示例）
                "code_patch:decode_topk80",
                "code_patch:surrogate_rect",
            ],
        )
        self.agent = SimpleAutoAgent()
        self.step = 0

    def _evaluate(self, agent: SimpleAutoAgent, seed: int) -> float:
        rng = random.Random(seed + int(agent.vth * 1000))
        base = 0.4 + agent.eta * 4.0 - abs(agent.vth - 0.6)
        return base + rng.random() * 0.05

    def run(self) -> Dict[str, object]:
        self.step += 1
        updated_agent, logs = self.meta.adapt(
            self.agent,
            step=self.step,
            evaluate_fn=lambda cand, s: self._evaluate(cand, s),
        )
        self.agent = updated_agent
        if logs:
            action, delta, reverted = _parse_log_delta(logs[-1])
        else:
            action, delta, reverted = ("none", 0.0, True)
        return {
            "reward": delta,
            "energy_penalty": 0.0,
            "meta_action": action,
            "delta": delta,
            "reverted": reverted,
            "note": logs[-1] if logs else "no-meta",
            "positive_ratio": self.meta.positive_ratio(),
        }


def run_rl(
    episodes: int,
    *,
    iteration: int | None = None,
    dream_period: int = 0,
    energy_cfg: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Run a short RL session; optionally enable dream every N daemon iterations."""
    use_dream = False
    dream_every = 0
    if isinstance(iteration, int) and dream_period and dream_period > 0:
        if (iteration % dream_period) == 0:
            use_dream = True
            dream_every = 1  # trigger dream rollout each episode this round
    cfg = energy_cfg or {}
    avg_return, success_rate, spikes = train_once(
        episodes=episodes,
        seed=int(time.time()) & 0xFFFF,
        use_dream=use_dream,
        dream_every=dream_every,
        homeo_on=cfg.get("homeo_on"),
        energy_target_low=cfg.get("energy_target_low"),
        energy_target_high=cfg.get("energy_target_high"),
        energy_ema=cfg.get("energy_ema"),
        energy_gamma=cfg.get("energy_gamma"),
        lambda_min=cfg.get("lambda_min"),
        lambda_max=cfg.get("lambda_max"),
    )
    prev = _RL_HISTORY.get("avg_return")
    delta_return = avg_return - prev if isinstance(prev, (int, float)) else avg_return
    _RL_HISTORY["avg_return"] = avg_return
    energy_penalty = spikes * ENERGY_COEFF_RL
    return {
        "task": "rl",
        "reward": delta_return,
        "energy_penalty": energy_penalty,
        "avg_return": avg_return,
        "success_rate": success_rate,
        "spikes": spikes,
        "meta_action": None,
        "delta": delta_return,
        "reverted": False,
        "cause_prob_self": min(1.0, max(0.0, success_rate)),
        "next_plan": f"继续 RL {episodes} 回合以提升成功率",
        "note": f"rl episodes={episodes}",
    }


def _summarize_domains(files: Dict[str, int], file_topics: Dict[str, str]) -> str:
    if not files:
        return "synthetic×?"
    summary = []
    for path, count in sorted(files.items(), key=lambda item: (-item[1], item[0])):
        topic = file_topics.get(path, "")
        topic_text = f"/{topic}" if topic and topic != "default" else ""
        summary.append(f"{Path(path).name}{topic_text}×{count}")
    return "，".join(summary)


def run_lm(num_lines: int, sampler: DomainSampler | None = None) -> Dict[str, object]:
    stats = train_lines(
        num_lines=num_lines,
        seed=int(time.time()) & 0xFFFF,
        sampler=sampler,
        valid_interval=max(50, num_lines // 2),
    )
    avg_loss = float(stats.get("avg_loss", 0.0))
    ppl = float(stats.get("ppl", 0.0))
    spikes = float(stats.get("avg_spikes", 0.0))
    valid_ppl = float(stats.get("valid_ppl", ppl))
    delta_ppl = float(stats.get("delta_ppl", 0.0))
    files = {str(k): int(v) for k, v in stats.get("files", {}).items()}
    file_topics = {str(k): str(v) for k, v in stats.get("file_topics", {}).items()}
    domain_text = _summarize_domains(files, file_topics)
    reward = -delta_ppl
    energy_penalty = spikes * ENERGY_COEFF_LM
    return {
        "task": "lm",
        "reward": reward,
        "energy_penalty": energy_penalty,
        "avg_loss": avg_loss,
        "ppl": ppl,
        "valid_ppl": valid_ppl,
        "delta_ppl": delta_ppl,
        "spikes": spikes,
        "meta_action": None,
        "delta": delta_ppl,
        "reverted": False,
        "domains_summary": domain_text,
        "corpus_path": (sampler.last_file() if sampler else None),
        "cause_prob_self": max(0.0, min(1.0, 0.5 + 0.1 * (1.0 / (1.0 + valid_ppl)))),
        "next_plan": f"继续 LM {num_lines} 行，巩固 Δppl {delta_ppl:+.3f}",
        "note": f"lm lines={num_lines}; files={domain_text}",
    }


def run_autoadapt(loop: AutoAdaptLoop) -> Dict[str, object]:
    payload = loop.run()
    payload.update(
        {
            "task": "autoadapt",
            "cause_prob_self": 0.5 + 0.3 * (0.0 if payload.get("reverted") else 1.0),
            "next_plan": "继续触发 MetaLearner，如果 Δ<=0 切换策略",
        }
    )
    # 将解释性说明写入 calibration_note，方便自述报告展示
    if payload.get("note"):
        payload["calibration_note"] = str(payload.get("note"))
    return payload


def run_post(
    length: int,
    *,
    topic_hint: str | None = None,
    temperature: float = 1.0,
    attempts: int = 3,
    post_thresholds: Dict[str, float] | None = None,
    sampler: "DomainSampler | None" = None,
    seed_text: str | None = None,
    allow_growth: bool = True,
) -> Dict[str, object]:
    """调用 SupervisedPost 进行解释性门控与自适应重试。"""
    thresholds = post_thresholds or {"overall": 0.62, "self_explain": 0.40}
    # 阶段 1：Retune（三段式解码重试由 SupervisedPost 内部执行）
    sp = SupervisedPost(attempts=attempts, thresholds=thresholds)
    res = sp.run(topic=topic_hint, max_len=length, seed_text=seed_text or "")
    text = str(res.get("text", ""))
    score = float(res.get("score", 0.0) or 0.0)
    details = res.get("details", {}) if isinstance(res.get("details"), dict) else {}
    read = float(details.get("readability", 0.0) or 0.0)
    ctx = float(details.get("context", 0.0) or 0.0)
    # 落地到 feed：复用 _write_feed（构造最小 GenerationResult）
    gen = GenerationResult(
        text=text,
        tokens_generated=len(text),
        spike_estimate=0.0,
        readability=read,
        context=ctx,
        notes=list(details.get("notes", [])) if isinstance(details.get("notes"), list) else [],
        attempts=[],
    )
    text_path = _write_feed(gen, topic_hint, length, temperature)
    actions_list = list(res.get("actions", []))
    trained = any("ntp:" in a for a in actions_list)
    patched = any("autopatch" in a for a in actions_list)
    # 先读取 attempts 再判断是否属于“重调参”
    attempts_used = int(res.get("attempts", 1) or 1)
    retuned = any(
        ("temperature" in a) or ("top_k" in a) or ("repeat_penalty" in a)
        for a in actions_list
    ) or (attempts_used > 1)
    best_act = str(res.get("best_action") or "baseline")
    note = f"post attempts={attempts_used} best={best_act} trained={trained} patched={patched} retuned={retuned} feed={Path(text_path).name}"

    # 阶段 2：Relearn —— overall<0.5 或 tokens<80 时触发；
    # 每批 600 行，最多 ~3000 行或 3 分钟；域优先匹配 topic。
    def _topic_biased_sampler(base: "DomainSampler | None", topic: str | None):
        if base is None or not topic:
            return base
        class _Wrap:
            def __init__(self, inner, hint: str):
                self.inner = inner
                self.hint = hint
                # expose attributes used by train_lines
                self.state_path = getattr(inner, "state_path", None)
            def next_lines(self, num_lines: int):
                # 尝试优先选择 topic 匹配的文件
                for _ in range(10):
                    it = self.inner.next_lines(num_lines)
                    try:
                        meta = self.inner.metadata_for(self.inner.last_file())
                        name_hit = False
                        try:
                            from pathlib import Path as _P
                            lf = self.inner.last_file()
                            if lf:
                                name_hit = (str(_P(lf).name).find(self.hint) >= 0)
                        except Exception:
                            name_hit = False
                        if (
                            isinstance(meta, dict)
                            and self.hint
                            and (
                                str(meta.get("topic", "")).find(self.hint) >= 0
                                or name_hit
                            )
                        ):
                            return it
                    except Exception:
                        pass
                return self.inner.next_lines(num_lines)
            def metadata_for(self, path=None):
                return self.inner.metadata_for(path)
            def last_file(self):
                return self.inner.last_file()
        return _Wrap(base, topic)

    tokens = len(text)
    relearned = False
    grown = False
    cooled = False
    if (score < 0.5) or (tokens < 80):
        try:
            from scripts.snn_text_lm import train_lines as _train_lines
            biased = _topic_biased_sampler(sampler, topic_hint)
            # 迭代小批训练：600 行/批，最多 5 批或 3 分钟
            import time as _t
            _start = _t.time()
            total = 0
            for _round in range(5):
                _ = _train_lines(600, sampler=biased, valid_interval=300)
                total += 600
                if (_t.time() - _start) >= 180.0:
                    break
            relearned = True
            seed_inject = "因为我们观察到目标不够清晰，所以我们先提出问题，再用例子推演，随后总结。"
            sp2 = SupervisedPost(attempts=max(3, attempts))
            res2 = sp2.run(topic=topic_hint, max_len=max(length, 200), seed_text=seed_inject)
            score2 = float(res2.get("score", 0.0) or 0.0)
            text2 = str(res2.get("text", ""))
            score2 = float(res2.get("score", 0.0) or 0.0)
            details2 = res2.get("details", {}) if isinstance(res2.get("details"), dict) else {}
            read2 = float(details2.get("readability", 0.0) or 0.0)
            ctx2 = float(details2.get("context", 0.0) or 0.0)
            # 覆盖 feed
            gen2 = GenerationResult(
                text=text2,
                tokens_generated=len(text2),
                spike_estimate=0.0,
                readability=read2,
                context=ctx2,
                notes=list(details2.get("notes", [])) if isinstance(details2.get("notes"), list) else [],
                attempts=[],
            )
            text_path = _write_feed(gen2, topic_hint, length, temperature)
            # 记录改参/补丁标记
            actions_list2 = list(res2.get("actions", []))
            trained = trained or any("ntp:" in a for a in actions_list2)
            patched = patched or any("autopatch" in a for a in actions_list2)
            retuned = retuned or any(("temperature" in a) or ("top_k" in a) or ("repeat_penalty" in a) for a in actions_list2)
            note += f" retrain_then_retry=True total_lines=1000 delta_overall={score2 - score:+.3f}"
            text = text2; score = score2; read = read2; ctx = ctx2
        except Exception as exc:
            note += f" retrain_then_retry=False err={exc}"

    # 阶段 3：Decode-Patch —— 仍不达标则尝试安全补丁 + 5 回合 A/B
    if (score < thresholds.get("overall", 0.62)) or (len(text) < 80):
        patch_note = ""
        try:
            from meta import autopatch as ap
            # 依次尝试 decode 补丁：trigram_on → repeat_115 → topk80
            candidates = ["decode:trigram_on", "decode:repeat_115", "decode:topk80"]
            for pid in candidates:
                ok, delta = ap.safe_apply_and_eval(pid, kind="post")
                d_score = (delta[0] if delta else 0.0) if delta else 0.0
                d_energy = (delta[1] if delta else 0.0) if delta else 0.0
                net = d_score - 0.02 * d_energy  # 采用轻度能耗折扣
                patch_note = f"patch={pid.split(':',1)[1]} status={'accepted' if ok else 'rolled_back'} Δscore={d_score:+.3f} Δenergy={d_energy:+.3f} net={net:+.3f}"
                if ok and net >= 0.0:
                    patched = True
                    break
            if not patched and not patch_note:
                patch_note = "patch=none"
        except Exception as exc:
            patch_note = f"patch=error err={exc}"

        # 再试一次生成以评估补丁效果
        sp3 = SupervisedPost(attempts=max(3, attempts), thresholds=thresholds)
        res3 = sp3.run(topic=topic_hint, max_len=max(length, 200))
        score3 = float(res3.get("score", 0.0) or 0.0)
        text3 = str(res3.get("text", ""))
        details3 = res3.get("details", {}) if isinstance(res3.get("details"), dict) else {}
        read3 = float(details3.get("readability", 0.0) or 0.0)
        ctx3 = float(details3.get("context", 0.0) or 0.0)
        gen3 = GenerationResult(
            text=text3,
            tokens_generated=len(text3),
            spike_estimate=0.0,
            readability=read3,
            context=ctx3,
            notes=list(details3.get("notes", [])) if isinstance(details3.get("notes"), list) else [],
            attempts=[],
        )
        text_path = _write_feed(gen3, topic_hint, length, temperature)
        note += f" decode_patch: {patch_note}"
        text = text3; score = score3; read = read3; ctx = ctx3

    # 阶段 4：Micro‑AutoGrow —— 仍不达标：尝试微扩容 + 再试；失败则回滚并 cooldown
    if (score < thresholds.get("overall", 0.62)) or (len(text) < 80):
        # 若上层声明冷却中，则跳过增长并标记 cooled
        if not allow_growth:
            cooled = True
        else:
            # 采用文件级 cooldown：runs/autogrow_state.json 记录最近触发时间
            import json as _json
            _state_file = Path("runs/autogrow_state.json")
            _state = {}
            try:
                _state = _json.loads(_state_file.read_text(encoding="utf-8")) if _state_file.exists() else {}
            except Exception:
                _state = {}
            import time as _t
            last = float(_state.get("last_ts", 0.0) or 0.0)
            if (_t.time() - last) < 600.0:  # 时间冷却期（兜底）
                cooled = True
            else:
                try:
                    # 通过继续学 + 再试，作为“微扩容”的近似方案（保持纯标准库）
                    from scripts.snn_text_lm import train_lines as _train_lines
                    biased = _topic_biased_sampler(sampler, topic_hint)
                    _ = _train_lines(600, sampler=biased, valid_interval=300)
                    sp4 = SupervisedPost(attempts=max(3, attempts), thresholds=thresholds)
                    res4 = sp4.run(topic=topic_hint, max_len=max(length, 220))
                    score4 = float(res4.get("score", 0.0) or 0.0)
                    text4 = str(res4.get("text", ""))
                    details4 = res4.get("details", {}) if isinstance(res4.get("details"), dict) else {}
                    read4 = float(details4.get("readability", 0.0) or 0.0)
                    ctx4 = float(details4.get("context", 0.0) or 0.0)
                    gen4 = GenerationResult(
                        text=text4,
                        tokens_generated=len(text4),
                        spike_estimate=0.0,
                        readability=read4,
                        context=ctx4,
                        notes=list(details4.get("notes", [])) if isinstance(details4.get("notes"), list) else [],
                        attempts=[],
                    )
                    text_path = _write_feed(gen4, topic_hint, length, temperature)
                    text = text4; score = score4; read = read4; ctx = ctx4
                    grown = True
                    _state["last_ts"] = _t.time()
                    _state_file.parent.mkdir(parents=True, exist_ok=True)
                    _state_file.write_text(_json.dumps(_state), encoding="utf-8")
                except Exception:
                    cooled = True
    return {
        "task": "post",
        "reward": score,
        "energy_penalty": 0.0,
        "readability": read,
        "context": ctx,
        "spikes": 0.0,
        "text_path": str(text_path),
        "meta_action": best_act if best_act != "baseline" else None,
        "delta": score,
        "reverted": False,
        "cause_prob_self": max(0.1, min(0.95, score)),
        "next_plan": (
            "若分数不足，将继续解码器微调与小步训练"
            if score < thresholds.get("overall", 0.62)
            else "得分良好，准备下一轮主题草稿"
        ),
        "note": note,
        "post_attempts": attempts_used,
        "post_best_action": best_act,
        "post_trained": trained,
        "post_patched": patched,
        "post_grown": grown,
        "post_cooldown": cooled,
        "post_relearned": relearned,
        "post_retuned": retuned,
        "post_threshold_overall": thresholds.get("overall", 0.62),
        "post_threshold_self": thresholds.get("self_explain", 0.40),
    }


def build_report_payload(iteration: int, metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "task": metrics.get("task", "unknown"),
        "episode": iteration,
        "avg_return": metrics.get("avg_return"),
        "success_rate": metrics.get("success_rate"),
        "ppl": metrics.get("ppl"),
        "valid_ppl": metrics.get("valid_ppl"),
        "energy": metrics.get("spikes"),
        "meta_action": metrics.get("meta_action"),
        "delta": metrics.get("delta"),
        "delta_ppl": metrics.get("delta_ppl"),
        "reverted": metrics.get("reverted"),
        "cause_prob_self": metrics.get("cause_prob_self"),
        "next_plan": metrics.get("next_plan"),
        "domains_summary": metrics.get("domains_summary"),
        "readability": metrics.get("readability"),
        "text_path": metrics.get("text_path"),
        "corpus_path": metrics.get("corpus_path") or find_train_corpus_from_config() or "",
        "relearned": metrics.get("post_relearned", False),
        "retuned": metrics.get("post_retuned", False),
        "patched": metrics.get("post_patched", False),
    }


def log_daemon_metrics(iteration: int, metrics: Dict[str, object]) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metric_a_value = metrics.get("avg_return")
    if metric_a_value is None:
        metric_a_value = metrics.get("avg_loss")
    if metric_a_value is None:
        metric_a_value = metrics.get("readability")
    metric_b_value = metrics.get("success_rate")
    if metric_b_value is None:
        metric_b_value = metrics.get("ppl")
    if metric_b_value is None:
        metric_b_value = metrics.get("valid_ppl")
    if metric_b_value is None:
        metric_b_value = metrics.get("context")
    append_daemon_row(
        {
            "timestamp": timestamp,
            "iteration": iteration,
            "task": metrics.get("task"),
            "reward": f"{metrics.get('reward', 0.0):.6f}",
            "energy_penalty": f"{metrics.get('energy_penalty', 0.0):.6f}",
            "metric_a": f"{metric_a_value:.6f}" if isinstance(metric_a_value, (int, float)) else "",
            "metric_b": f"{metric_b_value:.6f}" if isinstance(metric_b_value, (int, float)) else "",
            "spikes": f"{metrics.get('spikes', 0.0):.6f}",
            "meta_action": metrics.get("meta_action", ""),
            "delta": f"{metrics.get('delta', 0.0):.6f}",
            "reverted": metrics.get("reverted", False),
            "relearned": metrics.get("post_relearned", False),
            "retuned": metrics.get("post_retuned", False),
            "patched": metrics.get("post_patched", False),
            "grown": metrics.get("post_grown", False),
            "cooldown": metrics.get("post_cooldown", False),
            "note": metrics.get("note", ""),
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMSA background training daemon")
    parser.add_argument(
        "--loops",
        type=str,
        default="rl,lm,autoadapt,post",
        help="Comma-separated task list (subset of rl,lm,autoadapt,post).",
    )
    parser.add_argument("--poll-seconds", type=int, default=30, help="Sleep interval between tasks.")
    parser.add_argument("--rl-episodes", type=int, default=20, help="Episodes per RL call.")
    parser.add_argument("--rl-dream-period", type=int, default=0, help="Trigger dream on RL task every N daemon iterations (0=disabled).")
    parser.add_argument("--lm-lines", type=int, default=500, help="Lines per LM update.")
    parser.add_argument("--link-lm-lines", type=int, default=1500, help="Lines to train during post-link recovery (decode patch + LM + retry).")
    parser.add_argument("--corpus-path", type=str, default="", help="Optional explicit corpus text file path.")
    parser.add_argument("--post-len", type=int, default=220, help="Maximum characters per post generation.")
    parser.add_argument("--post-topic", type=str, default="", help="Optional topic hint for post generation.")
    parser.add_argument(
        "--post-fail-max",
        type=int,
        default=POST_FAIL_MAX,
        help=(
            "Max consecutive under-threshold posts before triggering linkage: "
            "code_patch:decode_trigram_on + train_lines(1500) + post retry."
        ),
    )
    parser.add_argument(
        "--post-cooldown",
        type=int,
        default=50,
        help="Post 微扩容动作的冷却回合数（同类动作间隔）。",
    )
    # Energy/homeostasis parameters for RL path
    parser.add_argument("--homeo", type=str, choices=["on", "off"], default="on", help="阈值自稳开关（RL）")
    parser.add_argument("--energy-target-low", type=float, default=0.012, help="能耗目标下界（RL）")
    parser.add_argument("--energy-target-high", type=float, default=0.020, help="能耗目标上界（RL）")
    parser.add_argument("--energy-ema", type=float, default=0.8, help="尖峰率 EMA 系数（RL）")
    parser.add_argument("--energy-gamma", type=float, default=0.5, help="λ(t) 调整步幅（RL）")
    parser.add_argument("--lambda-min", type=float, default=0.2, help="λ 下限（RL）")
    parser.add_argument("--lambda-max", type=float, default=3.0, help="λ 上限（RL）")
    parser.add_argument("--enable_autogrow", action="store_true", help="允许 Post 阶段微扩容")
    parser.add_argument("--resume", type=str, default="", help="预留：恢复策略（如 latest/best_return），当前仅记录")
    parser.add_argument("--post-threshold", type=float, default=0.62, help="Overall score threshold for supervised post.")
    parser.add_argument("--post-min-self", type=float, default=0.40, help="Self-explain score threshold for supervised post.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Optional limit for iterations (0 = run forever).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    allowed = [task.strip() for task in args.loops.split(",") if task.strip()]
    if not allowed:
        allowed = list(Scheduler.TASKS)
    Scheduler.TASKS = tuple(allowed)
    base_budgets = dict(Scheduler.BUDGETS)
    custom_budgets = {
        "rl": (args.rl_episodes, "episodes"),
        "lm": (args.lm_lines, "lines"),
        "autoadapt": (1, "steps"),
        "post": (args.post_len, "chars"),
    }
    Scheduler.BUDGETS = {
        task: custom_budgets.get(task, base_budgets.get(task, (1, "units")))
        for task in allowed
    }
    scheduler = Scheduler()
    auto_loop = AutoAdaptLoop()
    # 写入语料配置（可选）
    if args.corpus_path:
        write_corpus_config_for_path(args.corpus_path)
    lm_sampler = build_domain_sampler(corpus_path=args.corpus_path or None)
    iteration = read_last_iteration()
    post_fail_streak = 0
    last_grow_iter: int | None = None
    start_iteration = iteration
    try:
        while True:
            if args.max_iterations and (iteration - start_iteration) >= args.max_iterations:
                break
            selection = scheduler.select_next()
            task = str(selection)
            if task not in allowed:
                continue
            if task == "rl":
                episodes = int(getattr(selection, "budget", args.rl_episodes))
                metrics = run_rl(
                    episodes,
                    iteration=iteration + 1,
                    dream_period=int(args.rl_dream_period),
                    energy_cfg={
                        "homeo_on": (args.homeo == "on"),
                        "energy_target_low": float(args.energy_target_low),
                        "energy_target_high": float(args.energy_target_high),
                        "energy_ema": float(args.energy_ema),
                        "energy_gamma": float(args.energy_gamma),
                        "lambda_min": float(args.lambda_min),
                        "lambda_max": float(args.lambda_max),
                    },
                )
            elif task == "lm":
                num_lines = int(getattr(selection, "budget", args.lm_lines))
                metrics = run_lm(num_lines, sampler=lm_sampler)
            elif task == "post":
                length = int(getattr(selection, "budget", args.post_len))
                metrics = run_post(
                    length,
                    topic_hint=(args.post_topic or None),
                    temperature=1.0,
                    attempts=3,
                    post_thresholds={
                        "overall": float(args.post_threshold),
                        "self_explain": float(args.post_min_self),
                    },
                    sampler=lm_sampler,
                    allow_growth=(
                        bool(getattr(args, "enable_autogrow", False))
                        and (last_grow_iter is None or (iteration - last_grow_iter) >= int(args.post_cooldown))
                    ),
                )
                # 解释性门控：连续不达标则强制调度一次 lm 与 autoadapt
                passed = (
                    float(metrics.get("reward", 0.0)) >= float(args.post_threshold)
                    and float(metrics.get("cause_prob_self", 0.0)) >= float(args.post_min_self)
                )
                if not passed:
                    post_fail_streak += 1
                else:
                    post_fail_streak = 0
            else:
                metrics = run_autoadapt(auto_loop)
            scheduler.update(
                task,
                metrics["reward"],
                energy_penalty=metrics.get("energy_penalty", 0.0),
                note=metrics.get("note", ""),
            )
            iteration += 1
            metrics["task"] = task
            # 追加 post 解释性摘要到 self_report
            payload = build_report_payload(iteration, metrics)
            if task == "post":
                attempts_used = metrics.get("post_attempts", 1)
                best_act = metrics.get("post_best_action", "baseline")
                trained = metrics.get("post_trained", False)
                patched = metrics.get("post_patched", False)
                grown = metrics.get("post_grown", False)
                cooled = metrics.get("post_cooldown", False)
                relearned = metrics.get("post_relearned", False)
                retuned = metrics.get("post_retuned", False)
                payload["calibration_note"] = (
                    "观察→诊断→干预→结果→下一步："
                    f"本轮尝试 {attempts_used} 次解码调参（最佳 {best_act}），"
                    f"若未达标则按域继续学（已执行={bool(relearned)}），"
                    f"随后尝试 decode 补丁（已采纳={bool(patched)}），"
                    f"仍不足则微扩容（grown={bool(grown)}，cooldown={bool(cooled)}）。"
                )
            log_daemon_metrics(iteration, metrics)
            write_episode_report("runs/self_report.md", payload)

            # 记录增长发生的回合，用于基于迭代的冷却
            if task == "post" and bool(metrics.get("post_grown", False)):
                last_grow_iter = iteration

            # 若 post 连续不达标，触发“联动”：decode_trigram_on + 继续学 1500 行 + 再试一次 post
            if task == "post" and post_fail_streak >= int(args.post_fail_max):
                baseline_score = float(metrics.get("reward", 0.0) or 0.0)
                # 1) 强制 code_patch:decode_trigram_on（安全补丁流水线）
                patch_note = ""
                try:
                    from meta import autopatch as ap
                    ok, delta = ap.safe_apply_and_eval("decode:trigram_on", kind="post")
                    patch_note = f"patch=decode_trigram_on status={'accepted' if ok else 'rolled_back'} Δ={delta[0]:+0.3f}"
                except Exception as exc:
                    patch_note = f"patch=decode_trigram_on status=error err={exc}"

                # 2) 继续学：topic 相关域优先，训练 1500 行
                lm_lines = int(getattr(args, "link_lm_lines", 1500))
                lm_metrics = run_lm(lm_lines, sampler=lm_sampler)
                lm_metrics["task"] = "lm"
                lm_note = f"联动: lm{lm_lines} {patch_note}"
                scheduler.update(
                    "lm",
                    lm_metrics["reward"],
                    energy_penalty=lm_metrics.get("energy_penalty", 0.0),
                    note=lm_note,
                )
                iteration += 1
                log_daemon_metrics(iteration, lm_metrics)
                write_episode_report("runs/self_report.md", build_report_payload(iteration, lm_metrics))

                # 3) 再试一次 post 生成
                # 注入结构化提示以提升自解释与因果性，增大 Δoverall 概率
                seed_inject = (
                    "因为我们需要更清晰地梳理因果，所以我先陈述目标，"
                    "再给出步骤与例子，随后总结下一步。"
                )
                post_retry = run_post(
                    int(getattr(selection, "budget", args.post_len)),
                    topic_hint=(args.post_topic or None),
                    temperature=1.0,
                    attempts=4,
                    post_thresholds={
                        "overall": float(args.post_threshold),
                        "self_explain": float(args.post_min_self),
                    },
                    sampler=lm_sampler,
                    seed_text=seed_inject,
                )
                post_retry["task"] = "post"
                delta_overall = float(post_retry.get("reward", 0.0) or 0.0) - baseline_score
                # 记录“联动”事件到调度器 CSV
                scheduler.update(
                    "link",
                    delta_overall,
                    energy_penalty=0.0,
                    note=f"联动: trigram_on + lm{lm_lines} + post_retry Δoverall={delta_overall:+.3f}",
                )
                iteration += 1
                log_daemon_metrics(iteration, post_retry)
                write_episode_report("runs/self_report.md", build_report_payload(iteration, post_retry))
                # 重置 streak
                post_fail_streak = 0
            time.sleep(max(0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Daemon stopped by user.")


if __name__ == "__main__":
    main()
