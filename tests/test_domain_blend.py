from pathlib import Path
from tools.corpus import DomainSampler


def test_high_reward_domain_selected_more(tmp_path):
    # Prepare three small files
    f1 = tmp_path / "d1.txt"
    f2 = tmp_path / "d2.txt"
    f3 = tmp_path / "d3.txt"
    for i, fp in enumerate([f1, f2, f3], 1):
        fp.write_text("\n".join([f"line {i}-{j}" for j in range(50)]), encoding="utf-8")

    state_json = tmp_path / "sampler_state.json"
    sampler = DomainSampler(state_json=state_json, split="train", window=10, ucb_c=0.0, reward_alpha_ppl=1.0, reward_beta_explain=0.0)
    for fp in [f1, f2, f3]:
        sampler.register(fp)
    # Inject reward history: make f1 highest, f2 medium, f3 low
    meta = sampler._files  # type: ignore[attr-defined]
    meta[str(f1.resolve())]["reward_history"] = [0.8] * 10
    meta[str(f2.resolve())]["reward_history"] = [0.3] * 10
    meta[str(f3.resolve())]["reward_history"] = [-0.1] * 10
    # mark as explored to enable UCB path
    meta[str(f1.resolve())]["samples"] = 10
    meta[str(f2.resolve())]["samples"] = 10
    meta[str(f3.resolve())]["samples"] = 10
    sampler._state["files"] = dict(meta)  # type: ignore[attr-defined]
    # Compare UCB scores directly
    total = 100
    s1 = sampler._ucb_score(meta[str(f1.resolve())], total)  # type: ignore[attr-defined]
    s2 = sampler._ucb_score(meta[str(f2.resolve())], total)  # type: ignore[attr-defined]
    s3 = sampler._ucb_score(meta[str(f3.resolve())], total)  # type: ignore[attr-defined]
    assert s1 >= s2 >= s3
