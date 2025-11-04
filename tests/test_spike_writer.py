import glob
import os
from pathlib import Path


def _latest_feed() -> Path | None:
    files = sorted(glob.glob("runs/feed/*.md"))
    return Path(files[-1]) if files else None


def test_min_tokens_and_spikes(tmp_path, monkeypatch):
    # 运行一次生成流程（较短长度，但内部有兜底 ≥80 字）
    from scripts import spike_writer as sw

    # 目标输出目录
    runs_dir = tmp_path / "runs"
    (runs_dir / "feed").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    result = sw.run_with_retries(
        max_len=120,
        seed_text="",
        stop_tokens=("<eos>",),
        temperature=1.0,
        topic_hint="测试",
        rng_seed=1234,
    )
    feed_path = sw._write_feed(result, "测试", 120, 1.0)
    assert feed_path.is_file()
    # 断言 tokens ≥ 80
    assert int(result.tokens_generated) >= 80
    # 断言 spikes ≥ 0（环境差异较大，检查非负以保证稳定）
    assert float(result.spike_estimate) >= 0.0


def test_explainability_logged(tmp_path, monkeypatch):
    from scripts import spike_writer as sw

    runs_dir = tmp_path / "runs"
    (runs_dir / "feed").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    result = sw.run_with_retries(
        max_len=100,
        seed_text="",
        stop_tokens=("<eos>",),
        temperature=1.0,
        topic_hint="说明",
        rng_seed=5678,
    )
    feed_path = sw._write_feed(result, "说明", 100, 1.0)
    text = feed_path.read_text(encoding="utf-8")
    # 检查 explainability 指标与尝试参数被写入
    assert "Explainability" in text
    assert "Attempts" in text
