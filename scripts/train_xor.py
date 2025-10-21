"""CLI entry point for training the XOR SNN prototype."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn.model import train_xor


def main() -> None:
    final_acc = train_xor()
    assert final_acc >= 0.9, "XOR accuracy target not met."


if __name__ == "__main__":
    main()
