"""单文件版本的 XOR 脉冲网络训练入口。"""

from snn.model import train_xor
from tools.logger import get_logger, setup_logging


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)
    final_accuracy = train_xor()
    logger.info("脚本执行完成，最终准确率 %.3f", final_accuracy)
    assert final_accuracy >= 0.9, "XOR 准确率未达到 0.9 的目标。"
