"""Utilities for loading and featurising the synthetic Chinese mini corpus."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tools.config import find_train_corpus_from_config
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


@dataclass
class MiniCorpusRecord:
    text: str
    domain: str
    intent: str
    tone: str
    urgency: str


@dataclass
class MiniCorpusStats:
    vocab_size: int
    avg_length: float
    positive_ratio: float
    size: int


@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    eos_id: int
    unk_id: int

    def encode(self, text: str, *, append_eos: bool = True) -> List[int]:
        tokens = [self.stoi.get(ch, self.unk_id) for ch in text]
        if append_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: Sequence[int]) -> str:
        chars: List[str] = []
        for idx in ids:
            if 0 <= idx < len(self.itos):
                token = self.itos[idx]
                if token not in {"<pad>", "<eos>", "<unk>"}:
                    chars.append(token)
        return "".join(chars)

    @property
    def size(self) -> int:
        return len(self.itos)


def load_minicorpus(
    path: str | Path,
    *,
    max_sentences: int | None = None,
    seed: int | None = None,
) -> List[MiniCorpusRecord]:
    dataset_path = Path(path)
    records: List[MiniCorpusRecord] = []
    if not dataset_path.exists():
        # 回退：尝试从全局配置加载 HF 展开的 JSONL
        cfg_path = find_train_corpus_from_config()
        if cfg_path and str(cfg_path).endswith("train.txt"):
            # 使用可读文本，按双换行分隔；构造最小字段
            text_file = Path(cfg_path)
            text = text_file.read_text(encoding="utf-8")
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            for ch in chunks:
                # 简单构造领域/语气等标签
                domain = "学习"
                intent = "信息"
                tone = "积极" if (len(ch) % 2 == 0) else "中性"
                urgency = "中"
                records.append(
                    MiniCorpusRecord(text=ch.replace("\n", " "), domain=domain, intent=intent, tone=tone, urgency=urgency)
                )
        elif cfg_path and str(cfg_path).endswith("train.jsonl"):
            dataset_path = Path(cfg_path)
        else:
            raise FileNotFoundError(f"语料文件不存在且配置缺失: {dataset_path}")
    if dataset_path.exists() and not records:
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # 兼容原始 minicorpus.jsonl
                if all(k in data for k in ("text", "domain", "intent", "tone", "urgency")):
                    records.append(
                        MiniCorpusRecord(
                            text=data["text"],
                            domain=data["domain"],
                            intent=data["intent"],
                            tone=data["tone"],
                            urgency=data["urgency"],
                        )
                    )
                    continue
                # 兼容 HF JSONL：尝试将 question/answer 拼接为 text
                q = data.get("question") or data.get("query") or ""
                ans = data.get("answers") or data.get("answer") or ""
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if not isinstance(q, str):
                    q = str(q)
                if not isinstance(ans, str):
                    ans = str(ans)
                text = ("Q: " + q + " A: " + ans).strip()
                domain = "学习"
                intent = "信息"
                tone = "积极" if (len(text) % 2 == 0) else "中性"
                urgency = "中"
                records.append(MiniCorpusRecord(text=text, domain=domain, intent=intent, tone=tone, urgency=urgency))
    if max_sentences is not None and len(records) > max_sentences:
        rng = random.Random(seed)
        rng.shuffle(records)
        records = records[:max_sentences]
    return records


def build_char_vocab(
    records: Sequence[MiniCorpusRecord],
    *,
    max_size: int = 512,
    min_freq: int = 1,
) -> CharVocab:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(list(record.text))
    specials = ["<pad>", "<eos>", "<unk>"]
    frequent = [
        token
        for token, freq in counter.most_common(max(0, max_size - len(specials)))
        if freq >= min_freq and token not in specials
    ]
    itos = specials + frequent
    stoi = {token: idx for idx, token in enumerate(itos)}
    return CharVocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        eos_id=stoi["<eos>"],
        unk_id=stoi["<unk>"],
    )


def describe_corpus(records: Sequence[MiniCorpusRecord], vocab: CharVocab) -> MiniCorpusStats:
    if not records:
        return MiniCorpusStats(vocab_size=vocab.size, avg_length=0.0, positive_ratio=0.0, size=0)
    lengths = [len(record.text) for record in records]
    positives = sum(1 for record in records if record.tone == "积极")
    return MiniCorpusStats(
        vocab_size=vocab.size,
        avg_length=sum(lengths) / float(len(lengths)),
        positive_ratio=positives / float(len(records)),
        size=len(records),
    )


def iter_char_sequences(
    records: Sequence[MiniCorpusRecord],
    vocab: CharVocab,
    *,
    append_eos: bool = True,
) -> Iterator[Tuple[MiniCorpusRecord, List[int]]]:
    for record in records:
        yield record, vocab.encode(record.text, append_eos=append_eos)
