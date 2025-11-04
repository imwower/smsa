from pathlib import Path
from tools.reporter import write_episode_report


def test_conf_phrasing_and_empty_tag(tmp_path, monkeypatch):
    # 构造 self_model_metrics.csv 以触发话术映射
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    metrics = runs / "self_model_metrics.csv"
    metrics.write_text(
        "episode,spearman_rho,brier,window,conf_next,acc_next\n"
        "1,0.0,0.0,10,0.10,0.0\n"
        "2,0.0,0.0,10,0.50,0.5\n"
        "3,0.0,0.0,10,0.90,1.0\n",
        encoding="utf-8",
    )
    # 构造空样本 feed（tokens=0；包含一次尝试参数）
    feed = runs / "feed"
    feed.mkdir(exist_ok=True)
    feed_path = feed / "sample.md"
    feed_path.write_text(
        "# Spike Writer Output\n- Generated Tokens: 0\n- Spike Estimate: 0.00\n- Attempts:\n  - [try#1] | T=0.90 | top_k=48 | repeat_penalty=1.10 | trigram=0 | tokens=0 | spikes=0.00 | read=0.00 | ctx=0.00 | self=0.00 | overall=0.00 | pass=0\n",
        encoding="utf-8",
    )
    # 写入报告
    out = runs / "self_report.md"
    write_episode_report(
        out,
        {
            "task": "rl",
            "episode": 1,
            "avg_return": 0.0,
            "success_rate": 0.0,
            "energy": 0.0,
        },
    )
    # 写入一条 post，以触发 [空样本] 与解码参数打印
    write_episode_report(
        out,
        {
            "task": "post",
            "episode": 2,
            "text_path": str(feed_path),
            "energy": 0.0,
        },
    )
    text = out.read_text(encoding="utf-8")
    # 话术包含“置信”字样（由 compute_calibration 自动生成）
    assert "置信" in text or "Self-Model" in text
    # 标注空样本 & 参数行
    assert "空样本" in text
    assert "top_k=" in text
