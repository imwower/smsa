from scripts.train_gridworld import EpropGridAgent, GridWorldConfig


def test_energy_penalty_advantage():
    # 构造一个 agent，并比较两组相同 reward 下的优势函数（低能耗更优）
    cfg = GridWorldConfig()
    agent = EpropGridAgent(state_size=cfg.size * cfg.size, hidden_size=8, lambda_energy=0.02, seed=0)
    reward = 1.0
    probs = [0.25, 0.25, 0.25, 0.25]
    action = 0
    # 低尖峰计数 vs 高尖峰计数（平均化）
    low_counts = [0.1] * agent.hidden.n_out
    high_counts = [0.9] * agent.hidden.n_out
    # 模拟内部计算（不更新权重）
    hidden_dim = max(1, agent.hidden.n_out)
    adv_low = reward - agent.baseline - agent.lambda_energy * ((sum(low_counts) * agent.inner_steps) / float(hidden_dim))
    adv_high = reward - agent.baseline - agent.lambda_energy * ((sum(high_counts) * agent.inner_steps) / float(hidden_dim))
    assert adv_low > adv_high
