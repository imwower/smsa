"""日志工具模块，统一加载并提供日志接口。"""

from __future__ import annotations

import logging
import logging.config
import pathlib
import sys
from typing import Optional


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
