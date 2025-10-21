"""单文件版本的 XOR 脉冲网络训练入口。"""

import logging

from snn.model import train_xor


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    final_accuracy = train_xor()
    assert final_accuracy >= 0.9, "XOR 准确率未达到 0.9 的目标。"
