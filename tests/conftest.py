"""Test-suite hygiene.

1. The shared GitHub scan cache is module-level state, so it must be cleared
   between tests (fake repo objects can reuse memory addresses, which would
   otherwise collide with cached entries).
2. The suite is HERMETIC against the operator's ``.env``: agentics.config
   loads it into os.environ (override=False) — and it re-checks the file on
   EVERY ``Settings()`` construction, so deleting the keys before a test is
   not enough: the first Settings built inside the first test would reload
   the file mid-test and flip backend-sensitive assertions (live 2026-11-13:
   ``test_perplexity_only_mode_routes_everything_to_pplx`` got a
   HyperbrowserClient because the operator's ``BROWSER_BACKEND=hyperbrowser``
   re-entered os.environ after the fixture had deleted it). The fixture
   freezes the dotenv cache at the operator file's current mtime for the
   duration of each test — mid-test ``_load_dotenv()`` calls become no-ops —
   and removes every key the operator's .env (or the committed .env.example
   declares, guarding shell-exported variables on bare checkouts).
"""
import pytest

from agentics.config import PROJECT_ROOT


def _env_file_keys(name: str) -> list[str]:
    env_file = PROJECT_ROOT / name
    if not env_file.exists():
        return []
    keys: list[str] = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0].strip())
    return keys


def _operator_env_keys() -> list[str]:
    """Every key the lab's env files can inject (deduplicated, order kept)."""
    keys: list[str] = []
    for name in (".env.example", ".env"):     # example first: the contract
        for key in _env_file_keys(name):
            if key not in keys:
                keys.append(key)
    return keys


def _operator_env_mtime() -> float | None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return None
    try:
        return env_file.stat().st_mtime
    except OSError:
        return None


_LAB_ENV_KEYS = _operator_env_keys()


@pytest.fixture(autouse=True)
def _clean_scan_cache():
    from agentics.github_ops import issue_ops

    issue_ops.clear_scan_cache()
    yield
    issue_ops.clear_scan_cache()


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    import agentics.config as cfg

    # Freeze the dotenv loader: with _ENV_MTIME pinned to the operator
    # file's mtime, any mid-test _load_dotenv() sees "already loaded" and
    # returns without touching os.environ. Tests that exercise dotenv
    # loading point PROJECT_ROOT at their own tmp .env whose mtime differs,
    # so those keep working unchanged.
    monkeypatch.setattr(cfg, "_ENV_MTIME", _operator_env_mtime())
    for key in _LAB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
