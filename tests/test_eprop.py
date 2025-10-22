"""测试 DenseLIF 资格迹 (e-prop) 更新。"""

import unittest

from snn.dense import DenseLIF
from snn.lif import LIFParams


class TestEProp(unittest.TestCase):
    def test_eligibility_decay_and_accumulation(self) -> None:
        params = LIFParams(v_th=10.0, tau_m=5.0, tau_a=5.0, beta=0.0, refractory=1)
        layer = DenseLIF(
            n_in=1,
            n_out=1,
            params=params,
            surrogate_fn=lambda _: 0.3,
            eligibility_lambda=0.5,
        )
        layer.weights[0][0] = 0.0
        layer.bias[0] = 0.0
        layer.eligibility[0][0] = 0.8

        layer.step([1])

        expected = 0.8 * 0.5 + 0.3 * 1
        self.assertAlmostEqual(layer.eligibility[0][0], expected)


if __name__ == "__main__":
    unittest.main()
