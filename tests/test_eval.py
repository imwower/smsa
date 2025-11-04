from scripts.nightly_eval import main as nightly_main
from tools.change_summary import summarize_changes
from pathlib import Path


def test_nightly_and_change_summary(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    # 写入最小 CSV 日志
    (runs / "daemon.csv").write_text(
        "timestamp,iteration,task,reward,energy_penalty,metric_a,metric_b,spikes,meta_action,delta,reverted,relearned,retuned,patched,note\n"
        "2025-01-01T00:00:00Z,1,rl,0.1,0.0,0.1,0.2,10.0,,0.1,False,False,False,False,eta_e=0.04\n"
        "2025-01-01T00:30:00Z,2,lm,1.0,0.0,1.0,2.0,0.0,,1.0,False,False,False,False,inner_steps=14\n"
        "2025-01-01T01:00:00Z,3,post,0.7,0.0,0.7,0.7,0.0,,0.7,False,False,False,False,\n",
        encoding="utf-8",
    )
    (runs / "scheduler.csv").write_text(
        "timestamp,step,task,raw_reward,energy_penalty,penalized_reward,window_mean,window_std,normalized_reward,count,budget_value,budget_unit,note\n"
        "2025-01-01T00:00:00Z,1,rl,0.1,0.0,0.1,0,0,0,1,10,episodes,\n",
        encoding="utf-8",
    )
    (runs / "autopatch.log").write_text("ab_evaluate: 提升 (Δscore=+0.05, Δenergy=+0.0)", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    nightly_main(["--hours", "24"])  # 生成夜跑报表
    assert (runs / "nightly_report.md").is_file()

    summarize_changes(window=10)
    text = (runs / "change_summary.md").read_text(encoding="utf-8")
    assert "参数改动" in text
