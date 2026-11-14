"""List and stop lingering Hyperbrowser sessions across the WHOLE token pool.

Killed lab processes leave their cloud session active server-side until the
idle timeout; this tool stops them so a new run can create its session.
With the P18 pool one lingering session per token silently eats one of the
K parallel slots — so the default sweeps EVERY token in
configs/secrets.yaml ``hyperbrowser_api_keys`` (plus the singular
``hyperbrowser_api_key``), not just the first one.

Usage: python scripts/hb_session_cleanup.py [hb_... [hb_...]]
       (no args = sweep the whole pool from configs/secrets.yaml)
"""

import asyncio
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def tokens_from_secrets() -> list[str]:
    """Pool tokens + the singular fallback, de-duplicated, order kept."""
    cfg = REPO / "configs" / "secrets.yaml"
    tokens: list[str] = []
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        for key in (data.get("hyperbrowser_api_keys") or []):
            if key and str(key) not in tokens:
                tokens.append(str(key))
        single = data.get("hyperbrowser_api_key")
        if single and str(single) not in tokens:
            tokens.append(str(single))
    return tokens


async def sweep_token(token: str, label: str) -> int:
    """Stop every non-terminated session on ONE token; returns stop count."""
    from hyperbrowser import Hyperbrowser

    hb = Hyperbrowser(api_key=token)
    try:
        sessions = await asyncio.to_thread(hb.sessions.list)
    except Exception as exc:  # noqa: BLE001 — one dead token must not kill the sweep
        print(f"[{label}] list failed: {type(exc).__name__}: {exc}")
        return 0
    items = getattr(sessions, "data", None) or getattr(sessions, "sessions", None) or []
    if not items:
        return 0
    print(f"[{label}] {len(items)} session(s) on record")
    stopped = 0
    for s in items:
        sid = getattr(s, "id", "") or (s.get("id") if isinstance(s, dict) else "")
        state = (getattr(s, "state", "") or
                 (s.get("state") if isinstance(s, dict) else "") or "").lower()
        print(f"  {sid} state={state!r}")
        if sid and state not in ("stopped", "ended", "closed", "failed"):
            try:
                await asyncio.to_thread(hb.sessions.stop, sid)
                stopped += 1
                print(f"    -> stop requested")
            except Exception as exc:  # noqa: BLE001 — report, don't raise
                print(f"    -> stop failed: {type(exc).__name__}: {exc}")
    return stopped


async def main() -> int:
    tokens = sys.argv[1:] or tokens_from_secrets()
    if not tokens:
        print("no tokens given (args) and none found in configs/secrets.yaml")
        return 2
    total = 0
    for i, token in enumerate(tokens):
        label = f"token-{i + 1}/{len(tokens)}"
        total += await sweep_token(token, label)
    print(f"stopped {total} active session(s) across {len(tokens)} token(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
