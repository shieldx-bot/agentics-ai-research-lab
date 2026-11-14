"""Wave-3 #2 — static cascade routing (llm/router.py).

Contract under test:
* cascade is OFF by default and OFF unless a provider is configured;
* verdict/jury purposes can NEVER cascade (provenance + jury independence);
* a failed pool-side ask gets exactly ONE independent-provider retry,
  journalled truthfully (ok / failed), and never raises;
* the cascade client shares the judge session when the dual jury is
  configured, else builds its own cached session.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.llm import router as rt  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _settings(**kw):
    base = dict(llm_cascade_provider="", llm_cascade_purposes="research,plan,implement",
                judge_second_provider="", chatgpt_cookie_path="")
    base.update(kw)
    return SimpleNamespace(**base)


def _ctx(memory, settings, agent="a03"):
    return SimpleNamespace(agent_id=agent, memory=memory, settings=settings)


# --------------------------------------------------------------- provider
def test_cascade_off_by_default():
    assert rt.cascade_provider(_settings(), "research") == ""
    assert rt.cascade_provider(_settings(), "verify") == ""


def test_verdict_purposes_never_cascade():
    s = _settings(llm_cascade_provider="chatgpt")
    assert rt.cascade_provider(s, "verify") == ""
    assert rt.cascade_provider(s, "review") == ""
    assert rt.cascade_provider(s, "repair") == ""
    assert rt.cascade_provider(s, "defend") == ""
    assert rt.cascade_provider(s, "tournament") == ""
    assert rt.cascade_provider(s, "tournament_judge2") == ""
    assert rt.cascade_provider(s, "ideation") == ""
    # allowed science/plumbing purposes DO cascade
    assert rt.cascade_provider(s, "research") == "chatgpt"
    assert rt.cascade_provider(s, "plan") == "chatgpt"
    assert rt.cascade_provider(s, "implement") == "chatgpt"


def test_purpose_allowlist_is_configurable():
    s = _settings(llm_cascade_provider="chatgpt",
                  llm_cascade_purposes="research")
    assert rt.cascade_provider(s, "research") == "chatgpt"
    assert rt.cascade_provider(s, "plan") == ""


# ------------------------------------------------------------------ client
def test_get_cascade_client_requires_cookie(tmp_path, monkeypatch):
    s = _settings(llm_cascade_provider="chatgpt")
    assert rt.get_cascade_client(s) is None          # no cookie path at all
    s2 = _settings(llm_cascade_provider="chatgpt",
                   chatgpt_cookie_path=str(tmp_path / "missing.json"))
    assert rt.get_cascade_client(s2) is None         # file does not exist


def test_get_cascade_client_shares_judge_session(monkeypatch):
    s = _settings(llm_cascade_provider="chatgpt",
                  judge_second_provider="chatgpt")
    sentinel = object()
    monkeypatch.setattr("agentics.llm.chatgpt_client.get_judge_client",
                        lambda st: sentinel)
    assert rt.get_cascade_client(s) is sentinel


def test_try_cascade_off_is_none(tmp_path):
    m = SharedMemory(tmp_path / "mem.db")
    ctx = _ctx(m, _settings())
    assert asyncio.run(rt.try_cascade(ctx, "p", "concise", "research", "")) is None
    rows = m._conn.execute(
        "SELECT COUNT(*) FROM journal WHERE event='router_cascade'").fetchone()
    assert rows[0] == 0


# ------------------------------------------------------------------ retry
def _stub_client(answer="cascade says hi", raise_exc=None):
    async def ask(prompt, mode="concise", system=None, purpose="general"):
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(answer=answer, citations=["src1"])
    return SimpleNamespace(ask=ask)


def test_try_cascade_serves_and_journals(tmp_path, monkeypatch):
    m = SharedMemory(tmp_path / "mem.db")
    s = _settings(llm_cascade_provider="chatgpt")
    ctx = _ctx(m, s)
    monkeypatch.setattr(rt, "get_cascade_client", lambda st: _stub_client())
    out = asyncio.run(rt.try_cascade(ctx, "prompt", "concise", "plan", "sys"))
    assert out == ("cascade says hi", ["src1"])
    row = m._conn.execute(
        "SELECT detail FROM journal WHERE event='router_cascade' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    detail = json.loads(row[0])
    assert detail["outcome"] == "ok" and detail["purpose"] == "plan"


def test_try_cascade_failure_journals_and_returns_none(tmp_path, monkeypatch):
    m = SharedMemory(tmp_path / "mem.db")
    s = _settings(llm_cascade_provider="chatgpt")
    ctx = _ctx(m, s)
    boom = RuntimeError("cascade provider down")
    monkeypatch.setattr(rt, "get_cascade_client",
                        lambda st: _stub_client(raise_exc=boom))
    out = asyncio.run(rt.try_cascade(ctx, "prompt", "concise", "research", ""))
    assert out is None
    rows = m._conn.execute(
        "SELECT detail FROM journal WHERE event='router_cascade' AND "
        "detail LIKE '%failed%'").fetchall()
    assert rows, "the failed attempt must be journalled truthfully"


def test_try_cascade_empty_answer_is_none(tmp_path, monkeypatch):
    m = SharedMemory(tmp_path / "mem.db")
    s = _settings(llm_cascade_provider="chatgpt")
    ctx = _ctx(m, s)
    monkeypatch.setattr(rt, "get_cascade_client", lambda st: _stub_client(""))
    assert asyncio.run(rt.try_cascade(ctx, "p", "concise", "plan", "")) is None
