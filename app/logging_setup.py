"""Логирование: stdout + файл с ротацией."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LoggingConfig

FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(cfg: LoggingConfig) -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    formatter = logging.Formatter(FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if cfg.file:
        path = Path(cfg.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Библиотечный шум приглушаем.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
