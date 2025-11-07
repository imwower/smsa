"""Convert ChnSentiCorp (Chinese sentiment) into this project format.

Accepts TSV/CSV files with at least two columns: label and text.
Typical layouts:
  - TSV: label\treview
  - CSV: label,review  or  label,text

Produces train.jsonl and train.txt under the output directory.
Only uses the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


def _find_input_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in (".tsv", ".csv", ".jsonl"):
                files.append(p)
    return files


def _iter_rows(path: Path) -> Iterator[Tuple[str, str]]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                # Common HF fields: 'label', 'text' or 'review'
                label = obj.get("label")
                text = obj.get("text") or obj.get("review") or ""
                if text and label is not None:
                    yield str(label), str(text)
        return
    sep = "\t" if suf == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh, delimiter=sep)
        for row in reader:
            if not row:
                continue
            if len(row) >= 2:
                label = row[0].strip()
                text = row[1].strip()
                if not text and len(row) >= 3:
                    text = row[2].strip()
                if label and text:
                    yield label, text


def _tone_from_label(label: str) -> str:
    # Many variants use 1=positive, 0=negative or neutral.
    # We only need to distinguish positive (1) vs other (0) for cause_label.
    try:
        v = int(label)
        return "积极" if v == 1 else "中性"
    except Exception:
        pass
    # Try textual forms
    l = label.strip().lower()
    if l in ("pos", "positive", "+1"):
        return "积极"
    return "中性"


def convert_chnsenticorp(
    *, input_path: str, output_dir: str, min_length: int = 6
) -> Tuple[int, Path, Path]:
    src = Path(input_path).resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    txt_path = out_dir / "train.txt"

    files = _find_input_files(src)
    if not files:
        raise RuntimeError("No TSV/CSV files found for ChnSentiCorp")

    n = 0
    with jsonl_path.open("w", encoding="utf-8") as jh, txt_path.open(
        "w", encoding="utf-8"
    ) as th:
        for f in files:
            for label, text in _iter_rows(f):
                if len(text) < min_length:
                    continue
                record = {
                    "text": text,
                    "domain": "生活",  # product/service reviews → lifestyle by default
                    "intent": "情感",
                    "tone": _tone_from_label(label),
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
    p = argparse.ArgumentParser(description="Convert ChnSentiCorp to project corpus")
    p.add_argument("--input", required=True, help="Path to TSV/CSV file or folder")
    p.add_argument("--output", default="data/chnsenticorp", help="Output directory")
    p.add_argument("--min-length", type=int, default=6, help="Discard shorter samples")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n, jsonl_path, txt_path = convert_chnsenticorp(
        input_path=args.input,
        output_dir=args.output,
        min_length=args.min_length,
    )
    print(f"wrote {n} samples")
    print(f"jsonl: {jsonl_path}")
    print(f"text:  {txt_path}")


if __name__ == "__main__":
    main()
