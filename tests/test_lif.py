"""LIF 神经元组件行为测试。"""

import unittest

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate


def make_layer(tau_m: float, tau_a: float, v_th: float = 1.0) -> DenseLIF:
    params = LIFParams(
        v_th=v_th,
        tau_m=tau_m,
        tau_a=tau_a,
        beta=0.0,
        refractory=2,
    )
    layer = DenseLIF(
        n_in=1,
        n_out=1,
        params=params,
        surrogate_fn=fast_sigmoid_surrogate,
    )
    layer.weights[0][0] = 1.0
    layer.bias[0] = 0.0
    return layer


class TestLIFDynamics(unittest.TestCase):
    def test_membrane_update_scales_with_tau_m(self) -> None:
        fast = make_layer(tau_m=2.0, tau_a=10.0)
        slow = make_layer(tau_m=10.0, tau_a=10.0)
        fast.step([1])
        slow.step([1])
        self.assertGreater(fast.v[0], slow.v[0])

    def test_adaptation_scales_with_tau_a(self) -> None:
        layer_fast = make_layer(tau_m=2.0, tau_a=2.0)
        layer_slow = make_layer(tau_m=2.0, tau_a=10.0)
        layer_fast.weights[0][0] = 5.0
        layer_slow.weights[0][0] = 5.0
        layer_fast.step([1])
        layer_slow.step([1])
        self.assertGreater(layer_fast.a[0], layer_slow.a[0])

    def test_threshold_reset_and_refractory(self) -> None:
        layer = make_layer(tau_m=2.0, tau_a=10.0)
        layer.weights[0][0] = 5.0
        layer.step([1])
        self.assertEqual(layer.v[0], 0.0)
        self.assertEqual(layer.refractory[0], layer.params.refractory)


if __name__ == "__main__":
    unittest.main()
