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
from scripts.spike_writer import _write_feed, spike_generate
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
    "note",
]

LM_STATE_PATH = Path("runs/corpus_state.json")
ENERGY_COEFF_RL = 0.001
ENERGY_COEFF_LM = 0.0005
ENERGY_COEFF_POST = 0.0002

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
            actions=["eta_up", "eta_down", "vth_up", "vth_down"],
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
    return payload


def run_post(length: int, *, topic_hint: str | None = None, temperature: float = 1.0) -> Dict[str, object]:
    result = spike_generate(
        max_len=length,
        seed_text="",
        temperature=temperature,
        topic_hint=topic_hint,
    )
    text_path = _write_feed(result, topic_hint, length, temperature)
    reward = result.readability
    energy_penalty = result.spike_estimate * ENERGY_COEFF_POST
    return {
        "task": "post",
        "reward": reward,
        "energy_penalty": energy_penalty,
        "readability": result.readability,
        "context": result.context,
        "spikes": result.spike_estimate,
        "text_path": str(text_path),
        "meta_action": None,
        "delta": result.readability,
        "reverted": False,
        "cause_prob_self": max(0.1, min(0.9, result.readability)),
        "next_plan": "根据评分挑选优秀内容发布，并准备下一次主题草稿。",
        "note": f"post feed={text_path.name}",
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
                metrics = run_post(length)
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
            log_daemon_metrics(iteration, metrics)
            write_episode_report("runs/self_report.md", build_report_payload(iteration, metrics))
            time.sleep(max(0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Daemon stopped by user.")


if __name__ == "__main__":
    main()
