"""HuggingFace datasets 安装与展开工具（标准库 + datasets）。

默认下载 `suolyer/webqa`，将每个 split 展开为 JSONL 与可读文本。
可通过配置文件或 CLI 指定其它数据集。
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_DATASET = "suolyer/webqa"
DEFAULT_SPLITS = ("train",)
DEFAULT_OUTPUT = Path("data/hf/webqa")
CONFIG_PATH = Path("runs/datasets_config.json")


def _ensure_datasets() -> None:
    try:
        importlib.import_module("datasets")
        return
    except ImportError:
        pass
    # 安装 datasets
    cmd = [sys.executable, "-m", "pip", "install", "-q", "datasets"]
    subprocess.check_call(cmd)


def _load_cfg(config_path: Path = CONFIG_PATH) -> Tuple[str, Tuple[str, ...], Path]:
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            dataset = str(cfg.get("dataset") or DEFAULT_DATASET)
            splits = tuple(cfg.get("splits") or DEFAULT_SPLITS)
            out = Path(cfg.get("output_dir") or str(DEFAULT_OUTPUT))
            return dataset, splits, out
        except Exception:
            pass
    return DEFAULT_DATASET, DEFAULT_SPLITS, DEFAULT_OUTPUT


def _guess_readable_line(example: Mapping[str, object]) -> str:
    # 常见字段优先
    q = None
    a = None
    if "question" in example:
        q = example.get("question")
    elif "query" in example:
        q = example.get("query")
    if "answers" in example:
        ans = example.get("answers")
        if isinstance(ans, (list, tuple)) and ans:
            a = ans[0]
        elif isinstance(ans, str):
            a = ans
    elif "answer" in example:
        ans = example.get("answer")
        if isinstance(ans, (list, tuple)) and ans:
            a = ans[0]
        elif isinstance(ans, str):
            a = ans
    if isinstance(q, str) and isinstance(a, str):
        return f"Q: {q}\nA: {a}"

    # 次优：拼接 text/title
    title = example.get("title") if isinstance(example.get("title"), str) else None
    text = example.get("text") if isinstance(example.get("text"), str) else None
    if title or text:
        return (title or "") + ("\n" if title and text else "") + (text or "")

    # 兜底：输出 JSON
    return json.dumps(example, ensure_ascii=False)


def install_and_expand(
    dataset_id: Optional[str] = None,
    *,
    splits: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, int]:
    """安装并展开数据集到本地。

    返回各 split 写入的样本计数。
    """
    _ensure_datasets()
    from datasets import load_dataset  # type: ignore

    ds_id, cfg_splits, cfg_out = _load_cfg()
    ds_id = dataset_id or ds_id
    splits = tuple(splits or cfg_splits)
    out_dir = Path(output_dir or cfg_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, int] = {}
    ds = load_dataset(ds_id)
    for split in splits:
        if split not in ds:
            continue
        subset = ds[split]
        jsonl_path = out_dir / f"{split}.jsonl"
        txt_path = out_dir / f"{split}.txt"
        count = 0
        with jsonl_path.open("w", encoding="utf-8") as jh, txt_path.open(
            "w", encoding="utf-8"
        ) as th:
            for ex in subset:
                try:
                    line = _guess_readable_line(ex)
                except Exception:
                    line = json.dumps(ex, ensure_ascii=False)
                th.write(line.strip().replace("\r", "") + "\n\n")
                jh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
        written[split] = count
    return written


__all__ = ["install_and_expand", "CONFIG_PATH", "DEFAULT_DATASET"]

