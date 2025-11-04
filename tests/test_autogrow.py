from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate
from snn.policy import PolicyHead


def test_grow_and_prune_consistency():
    params = LIFParams(v_th=0.5, tau_m=8.0, tau_a=16.0, beta=0.35, refractory=2)
    hidden = DenseLIF(n_in=4, n_out=6, params=params, surrogate_fn=fast_sigmoid_surrogate, eligibility_lambda=0.9)
    policy = PolicyHead(n_in=hidden.n_out, n_actions=3, lr=0.05, seed=0)

    # 扩容 4 个隐藏单元
    hidden.grow_hidden(4)
    policy.grow_in(4)
    assert hidden.n_out == policy.n_in
    # 资格迹与权重尺寸一致
    assert len(hidden.bias) == hidden.n_out
    assert len(hidden.eligibility[0]) == hidden.n_out

    # 裁剪若干列
    idxs = [1, 3, 5]
    hidden.prune_hidden(idxs)
    policy.prune_in(idxs)
    assert hidden.n_out == policy.n_in
    assert len(hidden.bias) == hidden.n_out
