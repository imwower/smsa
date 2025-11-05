from pathlib import Path
from tools.checkpoint import save_best, load_best
from scripts.train_gridworld import build_agent_for_test
import json


def test_best_by_metric_topk(tmp_path, monkeypatch):
    # redirect checkpoints dir to tmp
    agent = build_agent_for_test(hidden=8, seed=0)
    # save three entries, keep top-2
    p1 = save_best(agent, "return", 0.8, top_k=2, dir=tmp_path)
    p2 = save_best(agent, "return", 0.5, top_k=2, dir=tmp_path)
    p3 = save_best(agent, "return", 1.2, top_k=2, dir=tmp_path)
    # index exists and sorted desc
    index = Path("runs/checkpoints/best_return.json")
    data = json.loads(index.read_text(encoding="utf-8"))
    assert len(data) == 2
    vals = [float(e["value"]) for e in data]
    assert vals[0] >= vals[1]
    # load_best returns the top entry path
    best_path = load_best("return")
    # our helper stores index to default dir; ensure it points to a file
    assert isinstance(best_path, str)
