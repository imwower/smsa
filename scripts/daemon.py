"""Background daemon that cycles through RL/LM/Meta tasks."""

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
from scripts.train_gridworld import train_once
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
    energy_penalty = spikes * 0.001
    return {
        "task": "rl",
        "reward": avg_return,
        "energy_penalty": energy_penalty,
        "avg_return": avg_return,
        "success_rate": success_rate,
        "spikes": spikes,
        "meta_action": None,
        "delta": 0.0,
        "reverted": False,
        "cause_prob_self": min(1.0, max(0.0, success_rate)),
        "next_plan": f"继续 RL {episodes} 回合以提升成功率",
        "note": f"rl episodes={episodes}",
    }


def run_lm(num_lines: int) -> Dict[str, object]:
    avg_loss, ppl, spikes = train_lines(num_lines=num_lines, seed=int(time.time()) & 0xFFFF)
    reward = -avg_loss
    energy_penalty = spikes * 0.0005
    return {
        "task": "lm",
        "reward": reward,
        "energy_penalty": energy_penalty,
        "avg_loss": avg_loss,
        "ppl": ppl,
        "spikes": spikes,
        "meta_action": None,
        "delta": 0.0,
        "reverted": False,
        "cause_prob_self": max(0.0, min(1.0, 0.5 + 0.1 * (1.0 / (1.0 + ppl)))),
        "next_plan": f"继续 LM {num_lines} 行以压低困惑度",
        "note": f"lm lines={num_lines}",
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


def build_report_payload(iteration: int, metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "task": metrics.get("task", "unknown"),
        "episode": iteration,
        "avg_return": metrics.get("avg_return"),
        "success_rate": metrics.get("success_rate"),
        "ppl": metrics.get("ppl"),
        "energy": metrics.get("spikes"),
        "meta_action": metrics.get("meta_action"),
        "delta": metrics.get("delta"),
        "reverted": metrics.get("reverted"),
        "cause_prob_self": metrics.get("cause_prob_self"),
        "next_plan": metrics.get("next_plan"),
    }


def log_daemon_metrics(iteration: int, metrics: Dict[str, object]) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metric_a_value = metrics.get("avg_return")
    if metric_a_value is None:
        metric_a_value = metrics.get("avg_loss")
    metric_b_value = metrics.get("success_rate")
    if metric_b_value is None:
        metric_b_value = metrics.get("ppl")
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
        default="rl,lm,autoadapt",
        help="Comma-separated task list (subset of rl,lm,autoadapt).",
    )
    parser.add_argument("--poll-seconds", type=int, default=30, help="Sleep interval between tasks.")
    parser.add_argument("--rl-episodes", type=int, default=20, help="Episodes per RL call.")
    parser.add_argument("--lm-lines", type=int, default=500, help="Lines per LM update.")
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
    Scheduler.BUDGETS = {
        task: base_budgets.get(task, (1, "units")) for task in allowed
    }
    scheduler = Scheduler()
    auto_loop = AutoAdaptLoop()
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
                metrics = run_rl(args.rl_episodes)
            elif task == "lm":
                metrics = run_lm(args.lm_lines)
            else:
                metrics = run_autoadapt(auto_loop)
            scheduler.update(task, metrics["reward"], energy_penalty=metrics.get("energy_penalty", 0.0))
            iteration += 1
            metrics["task"] = task
            log_daemon_metrics(iteration, metrics)
            write_episode_report("runs/self_report.md", build_report_payload(iteration, metrics))
            time.sleep(max(0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Daemon stopped by user.")


if __name__ == "__main__":
    main()
