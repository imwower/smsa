from scripts.train_gridworld import EpropGridAgent, GridWorldConfig


def test_lambda_adaptive_monotonic():
    cfg = GridWorldConfig()
    agent = EpropGridAgent(
        state_size=cfg.size * cfg.size,
        hidden_size=8,
        lambda_energy=0.5,
        energy_target_low=0.01,
        energy_target_high=0.02,
        energy_gamma=0.5,
        seed=0,
    )
    base = agent.lambda_energy
    # 高放电率：λ 上升（受上限限制）
    for _ in range(5):
        agent._update_lambda_from_norm(0.10)
    assert agent.lambda_energy >= base
    up = agent.lambda_energy
    # 低放电率：λ 下降（受下限限制）
    for _ in range(5):
        agent._update_lambda_from_norm(0.0)
    assert agent.lambda_energy <= up
