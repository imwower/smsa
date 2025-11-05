import os
from pathlib import Path

from tools.scheduler import Scheduler, SchedulerConfig
from scripts.train_gridworld import EpropGridAgent, GridWorldConfig


def test_lambda_monotonic_and_bounded():
    agent = EpropGridAgent(
        state_size=GridWorldConfig().size ** 2,
        hidden_size=16,
        energy_target_low=0.01,
        energy_target_high=0.02,
        energy_ema=0.0,
        energy_gamma=0.5,
        lambda_min=0.2,
        lambda_max=3.0,
        seed=0,
    )
    # Drive lambda up with consistently high normalized spikes
    ups = []
    for _ in range(10):
        agent._update_lambda_from_norm(0.08)
        ups.append(agent.lambda_energy)
    assert all(ups[i] <= ups[i + 1] + 1e-12 for i in range(len(ups) - 1))
    assert agent.lambda_energy <= 3.0 + 1e-9
    # Drive lambda down with low spikes
    downs = []
    for _ in range(10):
        agent._update_lambda_from_norm(0.0)
        downs.append(agent.lambda_energy)
    assert all(downs[i] >= downs[i + 1] - 1e-12 for i in range(len(downs) - 1))
    assert agent.lambda_energy >= 0.2 - 1e-9


def test_scheduler_energy_penalty_tracks_lambda(tmp_path):
    csv_path = tmp_path / "sched.csv"
    cfg = SchedulerConfig(csv_path=csv_path, state_path=tmp_path / "state.json")
    sched = Scheduler(config=cfg)
    # Simulate two updates with different lambda → energy_penalty
    lam_values = [0.4, 0.8]
    for lam in lam_values:
        sched.update("rl", reward=1.0, energy_penalty=lam, note=f"lambda={lam}")
    # Read back and compare last two rows' energy_penalty
    text = csv_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("timestamp")]  # drop header
    assert len(lines) >= 2
    last2 = lines[-2:]
    # columns: timestamp,step,task,raw_reward,energy_penalty,...
    e1 = float(last2[0].split(",")[4])
    e2 = float(last2[1].split(",")[4])
    assert e2 > e1
