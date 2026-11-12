"""Structured logging with per-agent prefixes (readable multi-thread logs)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from agentics.config import PROJECT_ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "agentics.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("github3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(agent_id: str = "system") -> logging.Logger:
    setup_logging()
    return logging.getLogger(agent_id)
