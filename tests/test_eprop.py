"""测试 DenseLIF 资格迹 (e-prop) 更新。"""

import unittest

from snn.dense import DenseLIF
from snn.lif import LIFParams


class TestEProp(unittest.TestCase):
    def test_eligibility_decays_by_lambda(self) -> None:
        params = LIFParams(v_th=5.0, tau_m=5.0, tau_a=5.0, beta=0.0, refractory=1)
        layer = DenseLIF(
            n_in=1,
            n_out=1,
            params=params,
            surrogate_fn=lambda _: 0.0,
            eligibility_lambda=0.5,
        )
        layer.weights[0][0] = 0.0
        layer.bias[0] = 0.0
        layer.eligibility[0][0] = 1.0

        layer.step([0])

        self.assertAlmostEqual(layer.eligibility[0][0], 0.5)

    def test_spike_increments_eligibility(self) -> None:
        params = LIFParams(v_th=5.0, tau_m=5.0, tau_a=5.0, beta=0.0, refractory=1)
        layer = DenseLIF(
            n_in=1,
            n_out=1,
            params=params,
            surrogate_fn=lambda _: 0.3,
            eligibility_lambda=0.5,
        )
        layer.weights[0][0] = 0.0
        layer.bias[0] = 0.0
        layer.eligibility[0][0] = 0.5

        layer.step([1])

        decayed = 0.5 * 0.5
        expected = decayed + 0.3 * 1
        self.assertAlmostEqual(layer.eligibility[0][0], expected)
        self.assertGreater(layer.eligibility[0][0], decayed)

    def test_bias_and_weight_eligibility_accumulate_over_steps(self) -> None:
        params = LIFParams(v_th=1.0, tau_m=5.0, tau_a=5.0, beta=0.0, refractory=1)
        layer = DenseLIF(
            n_in=1,
            n_out=1,
            params=params,
            surrogate_fn=lambda _: 0.25,
            eligibility_lambda=0.8,
        )
        layer.weights[0][0] = 0.0
        layer.bias[0] = 0.0
        contributions = []
        for step in range(3):
            layer.step([1])
            contributions.append(layer.eligibility[0][0])
        self.assertTrue(
            contributions[0] < contributions[1] < contributions[2],
            f"资格迹应在重复输入时持续累积：{contributions}",
        )
        self.assertAlmostEqual(
            layer.bias_eligibility[0],
            layer.eligibility[0][0],
            msg="单输入情况下偏置资格迹应与权重资格迹保持一致",
        )


if __name__ == "__main__":
    unittest.main()
