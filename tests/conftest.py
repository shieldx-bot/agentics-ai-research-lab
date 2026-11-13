"""Test-suite hygiene.

1. The shared GitHub scan cache is module-level state, so it must be cleared
   between tests (fake repo objects can reuse memory addresses, which would
   otherwise collide with cached entries).
2. The suite is HERMETIC against the operator's ``.env``: agentics.config
   loads it into os.environ at import time (override=False), so without this
   fixture a deployment's ``BROWSER_BACKEND=hyperbrowser`` /
   ``LLM_PROVIDER=perplexity`` / ``MAX_AGENTS=10`` silently flips tests that
   assert the defaults (a real incident: 8 session-host tests failed only on
   machines with a live .env). Every key the operator's .env sets is removed
   for the duration of each test and restored afterwards.
"""
import pytest

from agentics.config import PROJECT_ROOT


def _operator_env_keys() -> list[str]:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return []
    keys: list[str] = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0].strip())
    return keys


_LAB_ENV_KEYS = _operator_env_keys()


@pytest.fixture(autouse=True)
def _clean_scan_cache():
    from agentics.github_ops import issue_ops

    issue_ops.clear_scan_cache()
    yield
    issue_ops.clear_scan_cache()


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for key in _LAB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
