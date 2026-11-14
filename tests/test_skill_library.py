"""Wave-2 #1 — Voyager-style skill library.

Contract under test:
* skills come ONLY from supported settles; one row per source settle
  (re-distill refreshes, never duplicates);
* retrieval is token overlap + uses bump, degrades to '' on anything odd;
* distillation runs on the LOCAL router, stores the settled code VERBATIM,
  and never stores a skill without a name;
* cadence gate mirrors the reflection sweep.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.evolution import skills as sk  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _mem(tmp_path):
    return SharedMemory(tmp_path / "skills.db")


def _ctx(memory, cycle=4, **settings_kw):
    sbase = dict(skill_library_enabled=True,
                 skill_distill_every_cycles=4,
                 skill_max=120, local_llm_enabled=True)
    sbase.update(settings_kw)
    return SimpleNamespace(agent_id="a07", memory=memory, cycle=cycle,
                           settings=SimpleNamespace(**sbase),
                           journal=lambda *a, **k: None, meta={})


def _settled(tmp_path, claim="protocol claim about convergence rate",
             code="print('AGENTICS_RESULT: {\"rate\": 0.9}')"):
    m = _mem(tmp_path)
    hid = m.add_hypothesis("a01", claim,
                           {"metric": "rate", "op": ">=", "target": 0.5})
    m.update_hypothesis(hid, "supported")
    m.update_hypothesis_protocol_code(hid, code)
    return m, hid


# ------------------------------------------------------------------ memory
def test_add_and_refresh_one_skill_per_settle(tmp_path):
    m = _mem(tmp_path)
    sid = m.add_skill("rate_probe", "measures convergence rate", code="x=1",
                      metric="rate", source_hyp=7)
    again = m.add_skill("rate_probe_v2", "refreshed description",
                        code="x=2", metric="rate", source_hyp=7)
    assert sid == again                       # refresh, not duplicate
    skills = m.library_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "rate_probe_v2" and skills[0]["code"] == "x=2"


def test_add_skill_rejects_empty_name(tmp_path):
    assert _mem(tmp_path).add_skill("  ", "desc") == 0


def test_library_bounded(tmp_path):
    m = _mem(tmp_path)
    for i in range(10):
        m.add_skill(f"skill_{i}", "d", code="c", source_hyp=i, max_skills=5)
    assert len(m.library_skills(limit=100)) == 5


# --------------------------------------------------------------- retrieval
def test_relevant_skills_token_overlap(tmp_path):
    m = _mem(tmp_path)
    m.add_skill("convergence_probe",
                "measures convergence rate of iterative solvers",
                code="c", metric="rate")
    m.add_skill("weather_fetcher", "fetches weather api rows", code="c",
                metric="humidity")
    hits = sk.relevant_skills(m, "convergence rate of iterative solvers")
    assert len(hits) == 1 and hits[0]["name"] == "convergence_probe"
    assert m.library_skills()[0]["uses"] == 1   # retrieval bumps uses


def test_relevant_skills_empty_library(tmp_path):
    assert sk.relevant_skills(_mem(tmp_path), "anything here") == []


def test_skills_block_empty_when_no_hits(tmp_path):
    m = _mem(tmp_path)
    m.add_skill("weather_fetcher", "fetches weather api rows", code="c")
    assert sk.skills_block(m, "convergence rate of solvers") == ""


def test_skills_block_renders_hits(tmp_path):
    m = _mem(tmp_path)
    m.add_skill("convergence_probe",
                "measures convergence rate of iterative solvers",
                code="c", metric="rate")
    block = sk.skills_block(m, "convergence rate of iterative solvers")
    assert "Reusable skills" in block and "convergence_probe" in block


# -------------------------------------------------------------- distillation
def test_distill_stores_verbatim_code(tmp_path, monkeypatch):
    m, hid = _settled(tmp_path)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)
    captured = {}

    class _Resp:
        ok = True
        answer = ('{"name": "convergence_protocol", "description": '
                  '"prints AGENTICS_RESULT for convergence rate claims"}')

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        assert purpose == "skill_distill"
        captured["prompt"] = prompt
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    stored = asyncio.run(sk.distill_pending(_ctx(m)))
    assert stored == 1
    skill = m.skill_for_hyp(hid)
    assert skill["name"] == "convergence_protocol"
    assert skill["code"] == "print('AGENTICS_RESULT: {\"rate\": 0.9}')"
    assert "convergence rate" in captured["prompt"]


def test_distill_skips_unnamed_replies(tmp_path, monkeypatch):
    m, _hid = _settled(tmp_path)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    class _Resp:
        ok = True
        answer = '{"description": "no name in this reply"}'

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    assert asyncio.run(sk.distill_pending(_ctx(m))) == 0
    assert m.library_skills() == []


def test_distill_disabled_local(tmp_path):
    m, _hid = _settled(tmp_path)
    from agentics.llm import local_router
    ctx = _ctx(m, local_llm_enabled=False)
    # allowed_for -> False without monkeypatching (the real gate path)
    assert local_router.allowed_for(ctx.settings, "skill_distill") is False
    assert asyncio.run(sk.distill_pending(ctx)) == 0


def test_maybe_distill_cadence_and_flag(tmp_path, monkeypatch):
    m, _hid = _settled(tmp_path)
    calls = []

    async def fake_distill(ctx, limit=2):
        calls.append(ctx.cycle)
        return 0

    monkeypatch.setattr(sk, "distill_pending", fake_distill)
    sk.maybe_distill(_ctx(m, cycle=3))        # 3 % 4 != 0 -> skip
    assert calls == []
    sk.maybe_distill(_ctx(m, cycle=8))        # 8 % 4 == 0 -> run
    assert calls == [8]
    # disabled: never runs
    calls.clear()
    sk.maybe_distill(_ctx(m, cycle=8, skill_library_enabled=False))
    assert calls == []


def test_distill_ignores_already_distilled(tmp_path, monkeypatch):
    m, hid = _settled(tmp_path)
    m.add_skill("existing_skill", "already stored", code="c",
                source_hyp=hid)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    async def boom(prompt, settings=None, purpose="", system=""):
        raise AssertionError("no pending settle, router must not be called")

    monkeypatch.setattr(local_router, "ask", boom)
    assert asyncio.run(sk.distill_pending(_ctx(m))) == 0
