"""Convert THUCNews-style corpus into this project format.

Reads a THUCNews root folder where each subdirectory is a category and each
file contains a news article. Produces:

- train.jsonl with fields: text, domain, intent, tone, urgency
- train.txt with one sample per paragraph (double-newline separated)

Only uses the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Mapping


# Map THUCNews category name → project domain {生活, 职场, 出行, 学习}
_CAT_TO_DOMAIN: Dict[str, str] = {
    # Chinese category names
    "教育": "学习",
    "科技": "学习",
    "旅游": "出行",
    "汽车": "出行",
    "财经": "职场",
    "股票": "职场",
    "时政": "职场",
    "社会": "职场",
    "家居": "生活",
    "时尚": "生活",
    "娱乐": "生活",
    "星座": "生活",
    "彩票": "生活",
    "房产": "生活",
    "体育": "生活",
    # Common English/pinyin aliases (best-effort)
    "education": "学习",
    "tech": "学习",
    "technology": "学习",
    "travel": "出行",
    "auto": "出行",
    "car": "出行",
    "finance": "职场",
    "stock": "职场",
    "politics": "职场",
    "society": "职场",
    "home": "生活",
    "fashion": "生活",
    "entertainment": "生活",
    "constellation": "生活",
    "lottery": "生活",
    "house": "生活",
    "sport": "生活",
    "sports": "生活",
}


def _iter_thuc_files(root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (category_name, file_path) pairs recursively under ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        del dirnames  # not needed; we walk all
        d = Path(dirpath)
        if d == root:
            # skip root level files
            continue
        cat = d.name
        for fname in filenames:
            path = d / fname
            if not path.is_file():
                continue
            yield cat, path


def _iter_thuc_jsonl(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (category, text) from a JSONL dumped by datasets hub."""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Mapping[str, object] = json.loads(line)
            except Exception:
                continue
            # Attempt common field names: 'label' as category name or int; 'text' content
            cat_raw = obj.get("label") or obj.get("category") or obj.get("labels")
            text = obj.get("text") or obj.get("content") or obj.get("title")
            if isinstance(text, str) and text:
                cat = str(cat_raw) if cat_raw is not None else "新闻"
                yield cat, text


def _domain_from_category(cat: str) -> str:
    key = cat.strip().lower()
    # direct mapping
    if cat in _CAT_TO_DOMAIN:
        return _CAT_TO_DOMAIN[cat]
    if key in _CAT_TO_DOMAIN:
        return _CAT_TO_DOMAIN[key]
    # try to fuzzy-match by inclusion
    for k, v in _CAT_TO_DOMAIN.items():
        if k in cat or k in key:
            return v
    return "学习"


def _clean_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    # collapse excessive whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def convert_thucnews(
    *,
    input_dir: str,
    output_dir: str,
    max_samples: int | None = None,
    min_length: int = 10,
    seed: int | None = 0,
    update_config: bool = True,
) -> Tuple[int, Path, Path]:
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"THUCNews path not found: {root}")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    txt_path = out_dir / "train.txt"

    n = 0
    with jsonl_path.open("w", encoding="utf-8") as jh, txt_path.open(
        "w", encoding="utf-8"
    ) as th:
        if root.is_dir():
            samples: List[Tuple[str, Path]] = list(_iter_thuc_files(root))
            if not samples:
                raise RuntimeError("No files discovered under the THUCNews root.")
            if seed is not None:
                random.Random(seed).shuffle(samples)
            if max_samples is not None and max_samples > 0:
                samples = samples[:max_samples]
            for cat, path in samples:
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                text = _clean_text(raw)
                if len(text) < min_length:
                    continue
                domain = _domain_from_category(cat)
                record = {
                    "text": text,
                    "domain": domain,
                    "intent": "新闻",
                    "tone": "中性",
                    "urgency": "中",
                }
                jh.write(json.dumps(record, ensure_ascii=False) + "\n")
                th.write(text + "\n\n")
                n += 1
        elif root.is_file() and root.suffix.lower() == ".jsonl":
            for cat, text in _iter_thuc_jsonl(root):
                text = _clean_text(text)
                if len(text) < min_length:
                    continue
                domain = _domain_from_category(str(cat))
                record = {
                    "text": text,
                    "domain": domain,
                    "intent": "新闻",
                    "tone": "中性",
                    "urgency": "中",
                }
                jh.write(json.dumps(record, ensure_ascii=False) + "\n")
                th.write(text + "\n\n")
                n += 1
        else:
            raise RuntimeError("input must be THUCNews directory or a JSONL file")

    if update_config:
        # Best-effort: write datasets_config.json to point to the JSONL
        try:
            from tools.config import write_corpus_config_for_path

            write_corpus_config_for_path(str(jsonl_path))
        except Exception:
            pass

    return n, jsonl_path, txt_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert THUCNews to project corpus")
    p.add_argument("--input", required=True, help="THUCNews root directory")
    p.add_argument("--output", default="data/thucnews", help="Output directory")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples")
    p.add_argument("--min-length", type=int, default=10, help="Discard shorter samples")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    p.add_argument(
        "--no-config",
        action="store_true",
        help="Do not update runs/datasets_config.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n, jsonl_path, txt_path = convert_thucnews(
        input_dir=args.input,
        output_dir=args.output,
        max_samples=(args.max_samples or None),
        min_length=args.min_length,
        seed=args.seed,
        update_config=not args.no_config,
    )
    print(f"wrote {n} samples")
    print(f"jsonl: {jsonl_path}")
    print(f"text:  {txt_path}")


if __name__ == "__main__":
    main()
