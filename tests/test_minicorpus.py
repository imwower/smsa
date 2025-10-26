"""Mini corpus dataset and training smoke tests."""

from __future__ import annotations

import math
import unittest

from snn.datasets import build_char_vocab, iter_char_sequences, load_minicorpus
from snn.training.minicorpus import MiniCorpusConfig, train_minicorpus


class TestMiniCorpusDataset(unittest.TestCase):
    def test_loader_and_vocab_shapes(self) -> None:
        records = load_minicorpus("data/minicorpus.jsonl", max_sentences=120, seed=0)
        self.assertGreaterEqual(len(records), 60, "随机截取后仍应包含样本")
        vocab = build_char_vocab(records, max_size=80)
        self.assertLessEqual(vocab.size, 80)
        sequences = list(iter_char_sequences(records[:3], vocab))
        self.assertEqual(len(sequences), 3)
        for record, seq in sequences:
            self.assertGreater(len(seq), len(record.text))
            self.assertEqual(seq[-1], vocab.eos_id, "序列应以 <eos> 结尾")


class TestMiniCorpusTraining(unittest.TestCase):
    def test_training_loop_produces_samples(self) -> None:
        config = MiniCorpusConfig(
            dataset_path="data/minicorpus.jsonl",
            train_limit=40,
            val_limit=10,
            vocab_size=72,
            episodes=1,
            seed=321,
            meta_window=4,
            meta_min_delta=0.001,
        )
        result = train_minicorpus(config)
        self.assertTrue(math.isfinite(result.tail_nll))
        self.assertTrue(0.0 <= result.tail_cause_acc <= 1.0)
        self.assertTrue(0.0 <= result.tail_energy_mse <= 1.0)
        self.assertEqual(len(result.generated_samples), 3)
        for sample in result.generated_samples:
            self.assertIn(":", sample)
            self.assertGreater(len(sample.split(":", 1)[-1]), 0, "应生成可解释文本")


if __name__ == "__main__":
    unittest.main()
