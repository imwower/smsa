import time
from scripts.train_gridworld import build_agent_for_test


def _elig_cols(hidden) -> int:
    # eligibility is shape [n_in][n_out]
    return len(hidden.eligibility[0]) if hidden.eligibility else 0


def test_grow_and_prune_dimension_consistency():
    agent = build_agent_for_test(hidden=12, seed=123)
    h = agent.hidden
    p = agent.policy
    base_out = h.n_out
    h.grow_hidden(4)
    p.grow_in(4)
    assert h.n_out == base_out + 4
    assert p.n_in == h.n_out
    assert _elig_cols(h) == h.n_out
    # prune last two
    h.prune_hidden([h.n_out - 1, h.n_out - 2])
    p.prune_in([p.n_in - 1, p.n_in - 2])
    assert p.n_in == h.n_out
    assert _elig_cols(h) == h.n_out


def test_autogrow_cooldown_simple():
    agent = build_agent_for_test(hidden=10, seed=0)
    ok1, _ = agent.apply_modification("add_neuron")
    ok2, _ = agent.apply_modification("add_neuron")
    # cooldown blocks second immediate growth
    assert ok1 is True and ok2 is False
    # after cooldown period, should allow again
    time.sleep(1.1)
    ok3, _ = agent.apply_modification("add_neuron")
    assert ok3 is True

