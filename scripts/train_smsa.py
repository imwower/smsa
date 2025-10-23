"""Self-Modeling Spiking Agent (SMSA) 训练脚本入口。"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.training.smsa import RecoverySummary, train_smsa
from tools.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Modeling Spiking Agent 训练。")
    parser.add_argument("--episodes", type=int, default=200, help="训练回合数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="指标 CSV 的输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = get_logger(__name__)
    summary: RecoverySummary = train_smsa(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    logger.info(
        "SMSA 训练完成 baseline=%s dream=%s actual=%s",
        summary.baseline_recovery,
        summary.dream_recovery,
        summary.actual_recovery,
    )


if __name__ == "__main__":
    main()
