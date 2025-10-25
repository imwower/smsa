"""LIF 神经元组件行为测试。"""

import unittest

from snn.dense import DenseLIF
from snn.lif import LIFParams, fast_sigmoid_surrogate


def make_layer(
    tau_m: float,
    tau_a: float,
    v_th: float = 1.5,
    refractory: int = 2,
) -> DenseLIF:
    params = LIFParams(
        v_th=v_th,
        tau_m=tau_m,
        tau_a=tau_a,
        beta=0.0,
        refractory=refractory,
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
    def test_refractory_prevents_consecutive_spikes(self) -> None:
        layer = make_layer(tau_m=2.0, tau_a=5.0, v_th=0.5, refractory=2)
        layer.weights[0][0] = 5.0

        spike_1, _, _, _ = layer.step([1])
        spike_2, _, _, _ = layer.step([1])
        spike_3, _, _, _ = layer.step([1])
        spike_4, _, _, _ = layer.step([1])

        self.assertEqual(spike_1[0], 1, "首次应触发放电")
        self.assertEqual(spike_2[0], 0, "不应在不应期内重复放电")
        self.assertEqual(spike_3[0], 0, "不应期未结束仍不应放电")
        self.assertEqual(spike_4[0], 1, "不应期结束后应可再次放电")
        self.assertEqual(layer.v[0], 0.0, "触发后膜电位应被复位为 0")

    def test_membrane_resets_and_recovers_monotonically(self) -> None:
        layer = make_layer(tau_m=4.0, tau_a=8.0, v_th=0.5, refractory=1)
        layer.weights[0][0] = 5.0
        layer.step([1])
        self.assertEqual(layer.v[0], 0.0, "放电后膜电位应立即复位")
        # 释放不应期后，逐步衰减到 0，且过程单调
        values = []
        for _ in range(3):
            layer.step([0])
            values.append(layer.v[0])
        self.assertTrue(
            values[0] >= values[1] >= values[2],
            f"电位应单调衰减，得到序列 {values}",
        )

    def test_membrane_decay_is_slower_with_larger_tau_m(self) -> None:
        fast = make_layer(tau_m=2.0, tau_a=10.0)
        slow = make_layer(tau_m=10.0, tau_a=10.0)
        fast.v[0] = 1.0
        slow.v[0] = 1.0

        fast.step([0])
        slow.step([0])

        self.assertLess(fast.v[0], slow.v[0], "tau_m 更大时电位衰减应更慢")

    def test_adaptation_decay_is_slower_with_larger_tau_a(self) -> None:
        fast = make_layer(tau_m=2.0, tau_a=2.0)
        slow = make_layer(tau_m=2.0, tau_a=10.0)
        fast.a[0] = 1.0
        slow.a[0] = 1.0

        fast.step([0])
        slow.step([0])

        self.assertLess(fast.a[0], slow.a[0], "tau_a 更大时适应电流应衰减更慢")


if __name__ == "__main__":
    unittest.main()
