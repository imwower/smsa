"""XOR 脉冲网络训练脚本入口。"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.model import train_xor
from tools.logger import get_logger, setup_logging


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)
    final_acc = train_xor()
    logger.info("脚本执行完成，最终准确率 %.3f", final_acc)
    assert final_acc >= 0.9, "XOR 准确率未达到 0.9 的目标。"


if __name__ == "__main__":
    main()
