"""XOR 脉冲网络训练脚本入口。"""

from __future__ import annotations

import logging
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.model import train_xor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    final_acc = train_xor()
    assert final_acc >= 0.9, "XOR 准确率未达到 0.9 的目标。"


if __name__ == "__main__":
    main()
