"""Standalone XOR training entry that mirrors scripts/train_xor.py."""

from snn.model import train_xor


if __name__ == "__main__":
    final_accuracy = train_xor()
    assert final_accuracy >= 0.9, "XOR accuracy target not met."
