"""CLI for installing and expanding a HuggingFace dataset to local files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.datasets_loader import install_and_expand, CONFIG_PATH, DEFAULT_DATASET


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install and expand a datasets hub corpus")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="HF datasets id")
    p.add_argument("--splits", type=str, default="train", help="Comma-separated splits")
    p.add_argument("--output", type=str, default="", help="Output dir (defaults from config)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    splits = tuple([s.strip() for s in args.splits.split(",") if s.strip()])
    output = args.output or None
    written = install_and_expand(args.dataset, splits=splits, output_dir=output)
    print("written:", written)
    print(f"config path: {CONFIG_PATH}")


if __name__ == "__main__":
    main()

