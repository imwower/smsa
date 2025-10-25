"""日志工具模块，统一加载并提供日志接口。"""

from __future__ import annotations

import csv
import logging
import logging.config
import pathlib
import sys
from typing import Iterable, Mapping, Optional, Sequence


_CONFIG_LOADED = False
_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent / "logging.conf"
METRIC_FIELDNAMES = [
    "episode",
    "return",
    "success_rate",
    "spikes",
    "nll",
    "cause_acc",
    "meta_action",
    "delta",
    "reverted",
]


def setup_logging(config_path: Optional[pathlib.Path | str] = None) -> None:
    """加载日志配置文件，未找到时退回基础配置。"""
    global _CONFIG_LOADED
    if _CONFIG_LOADED:
        return

    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    if path.is_file():
        logging.config.fileConfig(
            path,
            defaults={"sys": sys},
            disable_existing_loggers=False,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    _CONFIG_LOADED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """返回指定名称的 Logger，自动保证配置已加载。"""
    if not _CONFIG_LOADED:
        setup_logging()
    return logging.getLogger(name)


class CsvLogger:
    """轻量级 CSV 记录器，负责写入表头并附加行。"""

    def __init__(self, path: pathlib.Path | str, fieldnames: Sequence[str]) -> None:
        self.path = pathlib.Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.is_file()
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not file_exists or self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row: Mapping[str, object]) -> None:
        payload = {key: row.get(key, "") for key in self.fieldnames}
        self._writer.writerow(payload)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del tb
        self.close()


class EpisodeMetricsLogger:
    """统一的训练指标记录器，负责写 CSV 并周期性打印摘要。"""

    def __init__(
        self,
        logger: logging.Logger,
        path: pathlib.Path | str,
        *,
        print_every: int = 10,
    ) -> None:
        self.logger = logger
        self.print_every = max(1, int(print_every))
        self._csv_logger = CsvLogger(path, METRIC_FIELDNAMES)

    def log(self, row: Mapping[str, object]) -> None:
        payload = {}
        for field in METRIC_FIELDNAMES:
            if field not in row:
                raise KeyError(f"缺少字段 {field}")
            payload[field] = row[field]
        self._csv_logger.log(payload)
        episode = int(payload["episode"])
        if episode % self.print_every == 0:
            self.logger.info(
                (
                    "Episode %03d return=%.3f success_rate=%.2f "
                    "spikes=%.1f nll=%.3f cause_acc=%.2f meta=%s delta=%.3f reverted=%s"
                ),
                episode,
                float(payload["return"]),
                float(payload["success_rate"]),
                float(payload["spikes"]),
                float(payload["nll"]),
                float(payload["cause_acc"]),
                payload["meta_action"],
                float(payload["delta"]),
                bool(payload["reverted"]),
            )

    def close(self) -> None:
        self._csv_logger.close()

    def __enter__(self) -> "EpisodeMetricsLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()
