"""Setup accounts: normalize raw cookie exports into accounts/account-NN.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
ACCOUNTS = PROJECT / "accounts"
PPLX_DOMAIN = "perplexity.ai"


def normalize(raw_path: Path) -> dict | None:
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(raw, dict) and "cookies" in raw:
        raw = raw["cookies"]
    if not isinstance(raw, list):
        return None
    cookies = [c for c in raw if PPLX_DOMAIN in str(c.get("domain", ""))]
    if not cookies:
        return None
    return {"cookies": cookies, "source": raw_path.name}


def freshness(cookies: list[dict]) -> tuple[str, str]:
    """Return (state, detail) based on session-token expiry."""
    token = next(
        (c for c in cookies if "session-token" in str(c.get("name", ""))), None
    )
    if not token:
        return "WARN", "no session-token cookie found"
    exp = token.get("expirationDate")
    if not exp:
        return "OK", "session cookie (no expiry recorded)"
    try:
        if isinstance(exp, str):
            exp_ts = time.mktime(time.strptime(exp[:10], "%Y-%m-%d"))
        else:
            exp_ts = float(exp)
    except (ValueError, TypeError):
        return "OK", "session cookie"
    days = (exp_ts - time.time()) / 86400
    if days < 0:
        return "EXPIRED", f"expired {-days:.0f} day(s) ago"
    return "OK", f"valid for another {days:.0f} day(s)"


def main() -> None:
    ACCOUNTS.mkdir(exist_ok=True)
    # gather every raw export that isn't already canonical
    candidates = [
        p for p in ACCOUNTS.glob("*.json")
        if not p.name.startswith("account-")
    ]
    used = []
    for raw in sorted(candidates):
        norm = normalize(raw)
        if norm is None:
            print(f"  skip {raw.name}: no perplexity cookies / not parseable")
            continue
        idx = 1
        while (ACCOUNTS / f"account-{idx:02d}.json").exists():
            idx += 1
        dest = ACCOUNTS / f"account-{idx:02d}.json"
        dest.write_text(json.dumps(norm["cookies"], indent=2, ensure_ascii=False),
                        encoding="utf-8")
        state, detail = freshness(norm["cookies"])
        print(f"  {raw.name} -> {dest.name}  [{state}] {detail}")
        used.append(dest.name)

    # freshness of existing canonical accounts
    for p in sorted(ACCOUNTS.glob("account-*.json")):
        if p.name in used:
            continue
        try:
            cookies = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(cookies, dict):
                cookies = cookies.get("cookies", [])
            state, detail = freshness(cookies)
            print(f"  {p.name}  [{state}] {detail}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {p.name}  [ERROR] {exc}")

    print("\nDone. Format: Cookie-Editor JSON export, only perplexity.ai cookies kept.")


if __name__ == "__main__":
    main()
