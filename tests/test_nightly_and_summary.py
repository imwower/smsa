from pathlib import Path
from scripts.nightly_eval import main as nightly_main
from tools.change_summary import summarize_changes


def test_nightly_and_change_summary(tmp_path, monkeypatch):
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    # Prepare minimal daemon/scheduler CSVs with headers
    daemon_csv = runs / "daemon.csv"
    daemon_csv.write_text(
        "timestamp,iteration,task,reward,energy_penalty,metric_a,metric_b,spikes,meta_action,delta,reverted,relearned,retuned,patched,grown,cooldown,note\n"
        "2025-01-01T00:00:01Z,1,rl,0.5,0.1,0.0,0.8,10.0,,0.5,False,False,False,False,False,False,rl test\n"
        "2025-01-01T00:00:02Z,2,lm,0.2,0.0,0.0,2.0,0.0,,0.2,False,False,False,False,False,False,lm test\n"
        "2025-01-01T00:00:03Z,3,post,0.6,0.0,0.6,0.5,0.0,,0.6,False,False,True,False,False,False,post test\n",
        encoding="utf-8",
    )
    sched_csv = runs / "scheduler.csv"
    sched_csv.write_text(
        "timestamp,step,task,raw_reward,energy_penalty,penalized_reward,window_mean,window_std,normalized_reward,count,budget_value,budget_unit,note\n"
        "2025-01-01T00:00:01Z,1,rl,0.5,0.1,0.4,0.0,1.0,0.0,1,10,episodes,n_hidden=40 grow\n",
        encoding="utf-8",
    )
    (runs / "autopatch.log").write_text("提升 Δscore=0.10\n回滚 Δscore=-0.05\n", encoding="utf-8")
    # Generate nightly report
    nightly_main(["--hours", "4"])  # writes runs/nightly_report.md
    assert (runs / "nightly_report.md").exists()
    # Generate change summary
    out = summarize_changes(window=10)
    text = Path(out).read_text(encoding="utf-8")
    assert "代码补丁" in text and "结构变更" in text

