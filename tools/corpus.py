"""Corpus monitoring and domain sampling utilities (standard library only)."""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = ["watch_corpus", "DomainSampler"]

_SPLITS = ("train", "valid", "test")
_DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _now_ts() -> float:
    return time.time()


def _load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"files": {}, "total_samples": 0}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"files": {}, "total_samples": 0}
    data.setdefault("files", {})
    data.setdefault("total_samples", 0)
    return data


def _freeze_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _ensure_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios 应包含 train/valid/test 三个值")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("ratios 总和须为正数")
    return tuple(value / total for value in ratios)  # type: ignore[return-value]


def _stable_split(path: str, ratios: Sequence[float]) -> str:
    normalized = _ensure_ratios(ratios)
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    value = int(digest, 16) / float(2**256)
    thresholds = [normalized[0], normalized[0] + normalized[1]]
    if value < thresholds[0]:
        return "train"
    if value < thresholds[1]:
        return "valid"
    return "test"


def _topic_from_path(path: Path) -> str:
    return path.parent.name or "default"


def _file_entry(path: Path, split: str) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "split": split,
        "topic": _topic_from_path(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "samples": 0,
        "pointer": 0,
        "delta_history": [],
        "last_delta": 0.0,
        "created": _now_ts(),
        "updated": _now_ts(),
    }


def _iter_existing(records: Mapping[str, Dict[str, object]]) -> Iterator[Dict[str, object]]:
    for entry in records.values():
        yield {
            "path": entry["path"],
            "split": entry["split"],
            "topic": entry.get("topic", "default"),
            "samples": entry.get("samples", 0),
            "delta_mean": _mean(entry.get("delta_history", [])),
        }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def watch_corpus(
    patterns: str | Sequence[str],
    state_json: str | Path,
    *,
    poll_seconds: float = 30.0,
    ratios: Sequence[float] = _DEFAULT_RATIOS,
) -> Iterator[Dict[str, object]]:
    """Yield file descriptors whenever new corpus files appear."""

    if isinstance(patterns, str):
        pattern_list = [patterns]
    else:
        pattern_list = list(patterns)
    if not pattern_list:
        raise ValueError("patterns 不能为空")
    state_path = Path(state_json)
    ratios = _ensure_ratios(ratios)
    state = _load_state(state_path)
    files: MutableMapping[str, Dict[str, object]] = state["files"]  # type: ignore[assignment]

    for record in _iter_existing(files):
        yield record

    while True:
        discovered: List[Dict[str, object]] = []
        for pattern in pattern_list:
            for match in glob.iglob(pattern):
                path = Path(match).resolve()
                if not path.is_file():
                    continue
                key = str(path)
                entry = files.get(key)
                if entry is None:
                    split = _stable_split(key, ratios)
                    entry = _file_entry(path, split)
                    files[key] = entry
                    discovered.append(entry)
                else:
                    stat = path.stat()
                    if entry.get("size") != stat.st_size:
                        entry["size"] = stat.st_size
                        entry["updated"] = _now_ts()
        if discovered:
            state["files"] = dict(files)
            _freeze_state(state_path, state)
            for entry in discovered:
                yield {
                    "path": entry["path"],
                    "split": entry["split"],
                    "topic": entry.get("topic", "default"),
                    "samples": entry.get("samples", 0),
                    "delta_mean": 0.0,
                }
        time.sleep(max(0.5, float(poll_seconds)))


class DomainSampler:
    """Domain-aware sampler that favours files with higher Δppl gains."""

    def __init__(
        self,
        state_json: str | Path,
        *,
        split: str = "train",
        window: int = 20,
        ucb_c: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"未知 split: {split}")
        if window <= 0:
            raise ValueError("window 需为正整数")
        if ucb_c < 0:
            raise ValueError("ucb_c 需为非负值")
        self.state_path = Path(state_json)
        self.split = split
        self.window = window
        self.ucb_c = ucb_c
        self._rng = random.Random(seed)
        self._active: Optional[str] = None
        self._state = _load_state(self.state_path)
        self._files: MutableMapping[str, Dict[str, object]] = self._state["files"]  # type: ignore[assignment]

    def _refresh(self) -> None:
        self._state = _load_state(self.state_path)
        self._files = self._state["files"]  # type: ignore[assignment]

    def _eligible_items(self) -> Dict[str, Dict[str, object]]:
        return {
            key: value
            for key, value in self._files.items()
            if value.get("split") == self.split
        }

    def register(self, path: str | Path, split: Optional[str] = None) -> str:
        resolved = str(Path(path).resolve())
        self._refresh()
        if resolved in self._files:
            return str(self._files[resolved].get("split", self.split))
        target_split = split or _stable_split(resolved, _DEFAULT_RATIOS)
        entry = _file_entry(Path(resolved), target_split)
        self._files[resolved] = entry
        self._state["files"] = dict(self._files)
        _freeze_state(self.state_path, self._state)
        return target_split

    def last_file(self) -> Optional[str]:
        """Return the most recent file path served by next_batch."""
        return self._active

    def metadata_for(self, path: str | Path | None = None) -> Dict[str, object]:
        """Return persisted metadata for a file (defaults to the last batch)."""
        target = path or self._active
        if not target:
            raise RuntimeError("当前无活跃文件，无法查询 metadata")
        key = str(Path(target).resolve())
        self._refresh()
        meta = self._files.get(key)
        if meta is None:
            raise KeyError(f"未知文件: {key}")
        return meta

    def _total_samples(self) -> int:
        total = int(self._state.get("total_samples", 0))
        if total <= 0:
            total = sum(int(meta.get("samples", 0)) for meta in self._files.values())
        return max(total, 1)

    def _ucb_score(self, meta: Mapping[str, object], total: int) -> float:
        history = meta.get("delta_history", [])
        mean_delta = _mean(history) if isinstance(history, list) else 0.0
        count = max(1, int(meta.get("samples", 0)))
        bonus = self.ucb_c * math.sqrt(2.0 * math.log(total + 1.0) / float(count))
        return mean_delta + bonus

    def _select_file(self) -> Tuple[str, Dict[str, object]]:
        self._refresh()
        items = self._eligible_items()
        if not items:
            raise RuntimeError(f"未找到 split={self.split} 的语料文件")
        unexplored = [
            (key, meta)
            for key, meta in items.items()
            if int(meta.get("samples", 0)) == 0
        ]
        if unexplored:
            index = self._rng.randrange(len(unexplored))
            return unexplored[index]
        total = self._total_samples()
        selected = max(items.items(), key=lambda item: self._ucb_score(item[1], total))
        return selected

    def _open_text(self, path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        return path.open("r", encoding="utf-8", errors="ignore")

    def next_batch(self, num_lines: int) -> Iterator[str]:
        if num_lines <= 0:
            raise ValueError("num_lines 需为正整数")
        key, meta = self._select_file()
        meta["samples"] = int(meta.get("samples", 0)) + 1
        self._state["total_samples"] = int(self._state.get("total_samples", 0)) + 1
        self._files[key] = meta
        self._state["files"] = dict(self._files)
        _freeze_state(self.state_path, self._state)
        self._active = key
        path = Path(key)
        pointer = int(meta.get("pointer", 0))

        def iterator() -> Iterator[str]:
            nonlocal pointer
            produced = 0
            try:
                while produced < num_lines:
                    emitted = 0
                    with self._open_text(path) as handle:
                        for idx, raw in enumerate(handle):
                            if idx < pointer:
                                continue
                            emitted += 1
                            pointer = idx + 1
                            produced += 1
                            yield raw.rstrip("\n")
                            if produced >= num_lines:
                                break
                    if emitted == 0:
                        if pointer == 0:
                            break
                        pointer = 0
                        continue
            finally:
                meta["pointer"] = pointer
                meta["updated"] = _now_ts()
                self._files[key] = meta
                self._state["files"] = dict(self._files)
                _freeze_state(self.state_path, self._state)

        return iterator()

    def record_delta(self, delta: float, path: str | Path | None = None) -> None:
        if path is None:
            if not self._active:
                raise RuntimeError("当前无活跃文件可记录 delta")
            key = self._active
        else:
            key = str(Path(path).resolve())
        self._refresh()
        meta = self._files.get(key)
        if meta is None:
            raise KeyError(f"未知文件: {key}")
        history = list(meta.get("delta_history", []))
        history.append(float(delta))
        if len(history) > self.window:
            history = history[-self.window :]
        meta["delta_history"] = history
        meta["last_delta"] = float(delta)
        meta["updated"] = _now_ts()
        self._files[key] = meta
        self._state["files"] = dict(self._files)
        _freeze_state(self.state_path, self._state)

    def describe(self) -> List[Dict[str, object]]:
        self._refresh()
        summary: List[Dict[str, object]] = []
        for key, meta in sorted(self._eligible_items().items()):
            summary.append(
                {
                    "path": key,
                    "topic": meta.get("topic"),
                    "samples": meta.get("samples", 0),
                    "delta_mean": _mean(meta.get("delta_history", [])),
                }
            )
        return summary
