"""统一配置加载器：读写 runs/datasets_config.json 并提供便捷查询。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_DIR = Path("runs")
CONFIG_PATH = CONFIG_DIR / "datasets_config.json"


def load_corpus_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_corpus_config_for_path(corpus_path: str) -> None:
    path = Path(corpus_path).resolve()
    cfg = {
        "dataset": "local",
        "splits": ["train"],
        "output_dir": str(path.parent),
        "inputs": [str(path)],
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def find_train_corpus_from_config() -> Optional[str]:
    cfg = load_corpus_config()
    inputs = cfg.get("inputs") or []
    if isinstance(inputs, list) and inputs:
        return str(inputs[0])
    out = cfg.get("output_dir")
    if out:
        train_txt = Path(out) / "train.txt"
        if train_txt.exists():
            return str(train_txt)
    return None


__all__ = [
    "CONFIG_PATH",
    "load_corpus_config",
    "write_corpus_config_for_path",
    "find_train_corpus_from_config",
]

