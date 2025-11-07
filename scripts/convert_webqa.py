"""Convert WebQA-like QA json/jsonl into this project format.

Accepts a JSONL file (one json object per line) or a JSON file containing a
list of objects. Each object should include at least ``question`` and
``answers`` (answers may be a list or a string).

Produces train.jsonl and train.txt under the output directory.
Only uses the standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Tuple


def _iter_examples(path: Path) -> Iterator[Mapping[str, object]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return
    # JSON array fallback
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        try:
            data = json.load(fh)
        except Exception:
            data = []
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj


def _qa_to_text(example: Mapping[str, object]) -> str:
    q = example.get("question") or example.get("query") or example.get("input") or ""
    ans = example.get("answers") or example.get("answer") or example.get("output") or ""
    # normalize
    if isinstance(ans, list) and ans:
        ans = ans[0]
    if not isinstance(q, str):
        q = str(q)
    if not isinstance(ans, str):
        ans = str(ans)
    q = q.strip().replace("\n", " ")
    a = ans.strip().replace("\n", " ")
    if not q and not a:
        return ""
    if q and a:
        return f"Q: {q}\nA: {a}"
    return q or a


def convert_webqa(*, input_path: str, output_dir: str, min_length: int = 6) -> Tuple[int, Path, Path]:
    src = Path(input_path).resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    txt_path = out_dir / "train.txt"

    n = 0
    with jsonl_path.open("w", encoding="utf-8") as jh, txt_path.open(
        "w", encoding="utf-8"
    ) as th:
        for ex in _iter_examples(src):
            text = _qa_to_text(ex)
            if len(text) < min_length:
                continue
            record = {
                "text": text,
                "domain": "学习",
                "intent": "问答",
                "tone": "中性",
                "urgency": "中",
            }
            jh.write(json.dumps(record, ensure_ascii=False) + "\n")
            th.write(text + "\n\n")
            n += 1

    try:
        from tools.config import write_corpus_config_for_path

        write_corpus_config_for_path(str(jsonl_path))
    except Exception:
        pass
    return n, jsonl_path, txt_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert WebQA json/jsonl to corpus")
    p.add_argument("--input", required=True, help="Path to train.json/train.jsonl")
    p.add_argument("--output", default="data/webqa", help="Output directory")
    p.add_argument("--min-length", type=int, default=6, help="Discard shorter samples")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n, jsonl_path, txt_path = convert_webqa(
        input_path=args.input, output_dir=args.output, min_length=args.min_length
    )
    print(f"wrote {n} samples")
    print(f"jsonl: {jsonl_path}")
    print(f"text:  {txt_path}")


if __name__ == "__main__":
    main()
