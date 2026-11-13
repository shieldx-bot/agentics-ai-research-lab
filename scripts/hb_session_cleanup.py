"""List and stop lingering Hyperbrowser sessions (free plan = 1 concurrent).

Killed lab processes leave their cloud session active server-side until the
idle timeout; this tool stops them so a new run can create its session.

Usage: python scripts/hb_session_cleanup.py [hb_...]
       (token defaults to configs/secrets.yaml hyperbrowser_api_keys)
"""

import asyncio
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def token_from_secrets() -> str:
    cfg = REPO / "configs" / "secrets.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        keys = data.get("hyperbrowser_api_keys") or []
        if keys:
            return str(keys[0])
    return ""


async def main() -> int:
    from hyperbrowser import Hyperbrowser

    token = sys.argv[1] if len(sys.argv) > 1 else token_from_secrets()
    if not token:
        print("no token given (arg 1) and none found in configs/secrets.yaml")
        return 2
    hb = Hyperbrowser(api_key=token)
    sessions = await asyncio.to_thread(hb.sessions.list)
    items = getattr(sessions, "data", None) or getattr(sessions, "sessions", None) or []
    print(f"{len(items)} session(s) on record")
    stopped = 0
    for s in items:
        sid = getattr(s, "id", "") or (s.get("id") if isinstance(s, dict) else "")
        state = (getattr(s, "state", "") or (s.get("state") if isinstance(s, dict) else "") or "").lower()
        print(f"  {sid} state={state!r}")
        if sid and state not in ("stopped", "ended", "closed", "failed"):
            try:
                await asyncio.to_thread(hb.sessions.stop, sid)
                stopped += 1
                print(f"    -> stop requested")
            except Exception as exc:  # noqa: BLE001 — report, don't raise
                print(f"    -> stop failed: {type(exc).__name__}: {exc}")
    print(f"stopped {stopped} active session(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
