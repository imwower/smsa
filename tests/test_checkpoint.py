from pathlib import Path
from tools.checkpoint import save_agent, load_agent
from scripts.train_gridworld import build_agent_for_test


def test_roundtrip_and_atomic(tmp_path):
    runs = tmp_path / "runs" / "checkpoints"
    runs.mkdir(parents=True, exist_ok=True)
    a = build_agent_for_test(hidden=8, seed=0)
    path = runs / "test.ckpt.gz"
    p = save_agent(a, path)
    assert Path(p).is_file()
    b = load_agent(p)
    # 关键子模块存在
    assert hasattr(b, "policy") and hasattr(b, "hidden")
    # 模拟中断：残留 .tmp 文件不会影响最终文件读取
    tmp = Path(p).with_suffix(Path(p).suffix + ".tmp")
    tmp.write_bytes(b"partial")
    b2 = load_agent(p)
    assert hasattr(b2, "policy")
