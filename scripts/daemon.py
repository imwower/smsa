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
    "note",
]

LM_STATE_PATH = Path("runs/corpus_state.json")
ENERGY_COEFF_RL = 0.001
ENERGY_COEFF_LM = 0.0005
ENERGY_COEFF_POST = 0.0002
POST_FAIL_MAX = 3  # 连续不达标次数阈值（默认 3）

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

    # 3) 回退：data/*.txt
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


def run_rl(episodes: int) -> Dict[str, object]:
    avg_return, success_rate, spikes = train_once(episodes=episodes, seed=int(time.time()) & 0xFFFF)
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
) -> Dict[str, object]:
    """调用 SupervisedPost 进行解释性门控与自适应重试。"""
    thresholds = post_thresholds or {"overall": 0.62, "self_explain": 0.40}
    sp = SupervisedPost(attempts=attempts, thresholds=thresholds)
    res = sp.run(topic=topic_hint, max_len=length)
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
    retuned = any(("temperature" in a) or ("top_k" in a) or ("repeat_penalty" in a) for a in actions_list)
    attempts_used = int(res.get("attempts", 1) or 1)
    best_act = str(res.get("best_action") or "baseline")
    note = f"post attempts={attempts_used} best={best_act} trained={trained} patched={patched} retuned={retuned} feed={Path(text_path).name}"

    # 自动继续学：overall<0.5 或 tokens<80 时触发，优先按 topic 选择域
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
                        if isinstance(meta, dict) and self.hint and str(meta.get("topic", "")).find(self.hint) >= 0:
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
    if (score < 0.5) or (tokens < 80):
        try:
            from scripts.snn_text_lm import train_lines as _train_lines
            biased = _topic_biased_sampler(sampler, topic_hint)
            _ = _train_lines(1000, sampler=biased)
            relearned = True
            # 再次生成
            sp2 = SupervisedPost(attempts=max(3, attempts))
            res2 = sp2.run(topic=topic_hint, max_len=max(length, 120))
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
            note += f" retrain_then_retry=True delta_overall={score2 - score:+.3f}"
            text = text2; score = score2; read = read2; ctx = ctx2
        except Exception as exc:
            note += f" retrain_then_retry=False err={exc}"
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
    parser.add_argument("--lm-lines", type=int, default=500, help="Lines per LM update.")
    parser.add_argument("--corpus-path", type=str, default="", help="Optional explicit corpus text file path.")
    parser.add_argument("--post-len", type=int, default=220, help="Maximum characters per post generation.")
    parser.add_argument("--post-topic", type=str, default="", help="Optional topic hint for post generation.")
    parser.add_argument("--post-fail-max", type=int, default=POST_FAIL_MAX, help="Max consecutive under-threshold posts before forced lm+autoadapt.")
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
                metrics = run_rl(episodes)
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
                relearned = metrics.get("post_relearned", False)
                retuned = metrics.get("post_retuned", False)
                payload["calibration_note"] = (
                    f"解释性尝试 {attempts_used} 次；最佳动作 {best_act}；"
                    f"继续学 {bool(relearned)}；改参 {bool(retuned)}；补丁 {bool(patched)}；训练 {bool(trained)}。"
                )
            log_daemon_metrics(iteration, metrics)
            write_episode_report("runs/self_report.md", payload)

            # 若 post 连续不达标，强制追加一次 lm 与一次 autoadapt
            if task == "post" and post_fail_streak >= int(args.post_fail_max):
                # LM
                num_lines = max(1000, int(args.lm_lines))
                lm_metrics = run_lm(num_lines, sampler=lm_sampler)
                scheduler.update("lm", lm_metrics["reward"], energy_penalty=lm_metrics.get("energy_penalty", 0.0), note=lm_metrics.get("note", ""))
                iteration += 1
                lm_metrics["task"] = "lm"
                log_daemon_metrics(iteration, lm_metrics)
                write_episode_report("runs/self_report.md", build_report_payload(iteration, lm_metrics))

                # AutoAdapt
                auto_metrics = run_autoadapt(auto_loop)
                scheduler.update("autoadapt", auto_metrics["reward"], energy_penalty=auto_metrics.get("energy_penalty", 0.0), note=auto_metrics.get("note", ""))
                iteration += 1
                auto_metrics["task"] = "autoadapt"
                log_daemon_metrics(iteration, auto_metrics)
                write_episode_report("runs/self_report.md", build_report_payload(iteration, auto_metrics))
                # 重置 streak
                post_fail_streak = 0
            time.sleep(max(0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Daemon stopped by user.")


if __name__ == "__main__":
    main()
