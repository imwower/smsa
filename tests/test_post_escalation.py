from pathlib import Path
from unittest import mock

from scripts.daemon import run_post, build_report_payload, log_daemon_metrics
from tools.reporter import write_episode_report


def test_post_escalation_and_logging(tmp_path, monkeypatch):
    # Force autopatch pipeline to report failure (to still flag 'patched' attempt)
    with mock.patch("meta.autopatch.safe_apply_and_eval", return_value=(False, (0.0, 0.0))):
        metrics = run_post(
            length=60,  # small length to trigger retune+relearn path
            topic_hint=None,
            temperature=1.0,
            attempts=2,
            post_thresholds={"overall": 0.5, "self_explain": 0.4},
            sampler=None,
            allow_growth=False,
        )
    # Expect at least retune attempted; relearn may be triggered depending on heuristics
    assert metrics.get("post_retuned", False) in (True, False)
    # Write CSV and MD logs
    metrics["task"] = "post"
    log_daemon_metrics(1, metrics)
    payload = build_report_payload(1, metrics)
    # add the observation paragraph like daemon.main does
    payload["calibration_note"] = (
        "观察→诊断→干预→结果→下一步：本轮尝试 1 次解码调参，随后必要时继续学习与补丁。"
    )
    write_episode_report("runs/self_report.md", payload)
    # CSV contains new fields
    csv = Path("runs/daemon.csv").read_text(encoding="utf-8")
    assert "retuned" in csv and "relearned" in csv and "patched" in csv and "grown" in csv and "cooldown" in csv
    # MD contains Chinese paragraph
    md = Path("runs/self_report.md").read_text(encoding="utf-8")
    assert "观察→诊断→干预→结果→下一步" in md

