"""Stop controller — owner decides when the infinite loop ends.

Create a file named STOP at the project root (CLI: `python -m agentics.cli stop`
or just `touch STOP`) and every agent will finish its current node, release its
claims and exit gracefully between cycles. Ctrl+C (SIGINT) behaves the same.
"""
from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from agentics.config import get_settings

_STOP_EVENT = asyncio.Event()
_SIGINT_COUNT = 0


def stop_file_path() -> Path:
    return get_settings().stop_file


def request_stop() -> None:
    """Owner requests shutdown — creates the STOP file."""
    stop_file_path().write_text(
        "stop requested\n", encoding="utf-8"
    )
    _STOP_EVENT.set()


def should_stop() -> bool:
    """True if owner requested stop (file present or SIGINT received)."""
    if _STOP_EVENT.is_set():
        return True
    return stop_file_path().exists()


def clear_stop() -> None:
    """Remove the STOP marker (used by tests / fresh start)."""
    p = stop_file_path()
    if p.exists():
        p.unlink()


async def wait_until_stopped(poll_seconds: float = 1.0) -> None:
    while not should_stop():
        await asyncio.sleep(poll_seconds)


def install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Turn SIGINT/SIGTERM into graceful shutdown (double Ctrl+C = hard exit)."""

    def _handler(sig_name: str) -> None:
        global _SIGINT_COUNT
        _SIGINT_COUNT += 1
        _STOP_EVENT.set()
        if _SIGINT_COUNT >= 2:
            raise KeyboardInterrupt(f"forced exit after double {sig_name}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig.name)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: _handler("SIG"))
