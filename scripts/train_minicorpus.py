"""Train the SelfModel on the synthetic minimal Chinese corpus."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.training.minicorpus import MiniCorpusConfig, train_minicorpus
from tools.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini Chinese corpus training entrypoint.")
    parser.add_argument("--dataset", type=str, default="data/minicorpus.jsonl", help="JSONL corpus path.")
    parser.add_argument("--episodes", type=int, default=35, help="Training epochs over the corpus.")
    parser.add_argument("--train-limit", type=int, default=2200, help="Maximum training sentences.")
    parser.add_argument("--val-limit", type=int, default=600, help="Validation sentences used for meta adaptation.")
    parser.add_argument("--vocab-size", type=int, default=384, help="Maximum vocabulary size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--meta-window", type=int, default=8, help="MetaLearner plateau window.")
    parser.add_argument("--meta-delta", type=float, default=0.01, help="MetaLearner plateau tolerance.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Log after every N training sentences (0 disables progress logs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = get_logger(__name__)
    config = MiniCorpusConfig(
        dataset_path=args.dataset,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        vocab_size=args.vocab_size,
        episodes=args.episodes,
        seed=args.seed,
        meta_window=args.meta_window,
        meta_min_delta=args.meta_delta,
        log_interval=args.log_interval,
    )
    result = train_minicorpus(config)
    logger.info(
        "MiniCorpus tail metrics: NLL=%.3f cause=%.2f energy=%.3f spikes=%.3f positives=%.2f",
        result.tail_nll,
        result.tail_cause_acc,
        result.tail_energy_mse,
        result.tail_spike_norm,
        result.positive_ratio,
    )
    for idx, sentence in enumerate(result.generated_samples, 1):
        logger.info("生成样例 %d: %s", idx, sentence)


if __name__ == "__main__":
    main()
