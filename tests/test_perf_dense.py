import time
from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate


def test_dense_step_perf_guard():
    # 宽松护栏：5k 步不超过 3 秒（环境相关，非精确基准）
    params = LIFParams(v_th=0.5, tau_m=8.0, tau_a=16.0, beta=0.35, refractory=2)
    layer = DenseLIF(n_in=16, n_out=32, params=params, surrogate_fn=fast_sigmoid_surrogate, eligibility_lambda=0.9)
    pre = [0] * layer.n_in
    t0 = time.perf_counter()
    steps = 5000
    for _ in range(steps):
        layer.step(pre)
    dt = time.perf_counter() - t0
    assert dt < 3.0
