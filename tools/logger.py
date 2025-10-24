"""日志工具模块，统一加载并提供日志接口。"""

from __future__ import annotations

import logging
import logging.config
import pathlib
import sys
from typing import Iterable, Mapping, Optional, Sequence

import csv


_CONFIG_LOADED = False
_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent / "logging.conf"


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
