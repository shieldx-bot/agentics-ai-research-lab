"""One-shot live probe: can this sandbox reach Perplexity via raw cookie
WebSocket? (Datacenter IPs are expected to get 403 — docs/LLM_PROVIDERS.md.)

Usage: python scripts/probe_local_cookie.py [account.json]
"""

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.llm.pool import load_account_file  # noqa: E402
from agentics.llm.perplexity_client import PerplexityClient  # noqa: E402


async def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "accounts" / "account-10.json")
    account = load_account_file(path)
    print(f"probe {path.name}: {len(account.cookies)} pplx cookies")
    client = PerplexityClient(account, timeout=60)
    try:
        resp = await client.ask("Reply with exactly one word: ok", mode="concise")
        print(f"ASK OK: answer={resp.answer[:80]!r} citations={len(resp.citations)}")
        return 0
    except Exception as exc:  # noqa: BLE001 — the probe reports, never raises
        print(f"ASK FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
