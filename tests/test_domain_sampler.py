import glob
from pathlib import Path
from tools.corpus import DomainSampler


def test_sampler_prefers_high_reward(tmp_path):
    # 构造三域文本
    data = tmp_path / "data"
    data.mkdir()
    a = data / "a.txt"
    b = data / "b.txt"
    c = data / "c.txt"
    a.write_text("因为 所以 因此 我 计划 行动 结果\n" * 50, encoding="utf-8")
    b.write_text("一般 文本 内容\n" * 50, encoding="utf-8")
    c.write_text("杂项\n" * 50, encoding="utf-8")

    state = tmp_path / "runs" / "corpus_state.json"
    sam = DomainSampler(state_json=state, split="train", window=20, ucb_c=0.2)
    sam.register(a, split="train")
    sam.register(b, split="train")
    sam.register(c, split="train")

    counts = {str(a): 0, str(b): 0, str(c): 0}
    # 模拟 300 次抽样并回写 Δppl：A > B > C
    for _ in range(300):
        _ = list(sam.next_lines(8))
        lf = sam.last_file()
        counts[lf] += 1
        if lf == str(a):
            sam.record_delta(0.08)
        elif lf == str(b):
            sam.record_delta(0.03)
        else:
            sam.record_delta(0.00)

    # 最高奖励域 a 的抽样次数最多
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    assert Path(top).name == "a.txt"
