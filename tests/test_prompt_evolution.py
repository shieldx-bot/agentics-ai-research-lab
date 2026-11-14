"""Wave-2 #2 — DSPy-lite prompt evolution.

Contract under test:
* the challenger pool is bounded and scored on real outcomes only;
* Thompson/explore serving degrades to the champion on every failure;
* an arm that beats the champion's win-rate by the margin is promoted
  through the ordinary store_role_prompt machinery (TTL/rework untouched);
* mutation runs on the LOCAL router only (never the cookie pool).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.evolution import prompts as pevo  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _mem(tmp_path):
    return SharedMemory(tmp_path / "pevo.db")


def _ctx(memory, **kw):
    base = dict(agent_id="a01", role="coder", memory=memory,
                settings=SimpleNamespace(
                    prompt_evolution_enabled=True, prompt_variant_pool=3,
                    prompt_explore_p=0.25, prompt_variant_promote_min=10,
                    prompt_variant_promote_edge=0.15,
                    local_llm_enabled=False),
                meta={}, journal=lambda *a, **k: None)
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------ memory
def test_variant_pool_is_bounded(tmp_path):
    m = _mem(tmp_path)
    for i in range(5):
        m.store_prompt_variant("coder", "code", f"variant mission {i} " * 20,
                               pool_size=3)
    pool = m.prompt_variants("coder", "code")
    assert len(pool) == 3                       # newest 3 armed
    assert pool[-1]["mission"].startswith("variant mission 4")


def test_store_variant_rejects_empty(tmp_path):
    m = _mem(tmp_path)
    assert m.store_prompt_variant("coder", "code", "  ") == 0


def test_outcome_recording(tmp_path):
    m = _mem(tmp_path)
    vid = m.store_prompt_variant("coder", "code", "x " * 150)
    m.record_prompt_variant_outcome(vid, True)
    m.record_prompt_variant_outcome(vid, False)
    arm = m.prompt_variants("coder", "code")[0]
    assert arm["uses"] == 2 and arm["wins"] == 1 and arm["losses"] == 1


def test_promotion_beats_champion(tmp_path):
    m = _mem(tmp_path)
    m.store_role_prompt("coder", "code", "champion mission", reason="cold")
    champ_id = m.get_role_prompt("coder", "code")["id"]
    # champion: 10 uses, 2 wins -> rate 0.2
    for _ in range(2):
        m.bump_role_prompt("coder", "code", success=True, rework_after=99)
    for _ in range(8):
        m.bump_role_prompt("coder", "code", success=False, rework_after=99)
    vid = m.store_prompt_variant("coder", "code", "better variant " * 30)
    for _ in range(10):
        m.record_prompt_variant_outcome(vid, True)   # rate 1.0
    assert m.promote_prompt_variant(vid, min_uses=10, edge=0.15) is True
    new = m.get_role_prompt("coder", "code")
    assert new["id"] != champ_id and new["mission"].startswith("better")
    assert new["reason"] == "prompt_evolved"
    assert m.prompt_variants("coder", "code") == []  # arm left the pool


def test_promotion_refused_without_margin(tmp_path):
    m = _mem(tmp_path)
    m.store_role_prompt("coder", "code", "champion mission", reason="cold")
    for _ in range(10):
        m.bump_role_prompt("coder", "code", success=True, rework_after=99)
    vid = m.store_prompt_variant("coder", "code", "equal variant " * 30)
    for _ in range(10):
        m.record_prompt_variant_outcome(vid, True)   # rate 1.0 == champion
    assert m.promote_prompt_variant(vid, min_uses=10, edge=0.15) is False


def test_promotion_refused_without_samples(tmp_path):
    m = _mem(tmp_path)
    m.store_role_prompt("coder", "code", "champion mission", reason="cold")
    vid = m.store_prompt_variant("coder", "code", "variant " * 50)
    m.record_prompt_variant_outcome(vid, True)
    assert m.promote_prompt_variant(vid, min_uses=10, edge=0.15) is False


# ----------------------------------------------------------------- serving
def test_disabled_evolution_serves_champion(tmp_path):
    m = _mem(tmp_path)
    ctx = _ctx(m, settings=SimpleNamespace(prompt_evolution_enabled=False))
    mission, vid = asyncio.run(pevo.maybe_variant(ctx, "coder", "code",
                                                  "champion"))
    assert (mission, vid) == ("champion", 0)


def test_explore_off_serves_champion(tmp_path):
    m = _mem(tmp_path)
    m.store_prompt_variant("coder", "code", "variant " * 100)
    ctx = _ctx(m, settings=SimpleNamespace(
        prompt_evolution_enabled=True, prompt_explore_p=0.0,
        prompt_variant_pool=3, local_llm_enabled=False))
    mission, vid = asyncio.run(pevo.maybe_variant(ctx, "coder", "code",
                                                  "champion"))
    assert (mission, vid) == ("champion", 0)


def test_explore_full_can_serve_variant(tmp_path):
    m = _mem(tmp_path)
    vid = m.store_prompt_variant("coder", "code", "variant mission " * 40)
    ctx = _ctx(m, settings=SimpleNamespace(
        prompt_evolution_enabled=True, prompt_explore_p=1.0,
        prompt_variant_pool=3, local_llm_enabled=False))
    mission, served = asyncio.run(pevo.maybe_variant(ctx, "coder", "code",
                                                     "champion"))
    assert served == vid and mission.startswith("variant mission")
    # resolve_system_prompt stashes the served id for the feedback path
    ctx.meta["_prompt_variant_id"] = served
    pevo.note_variant_feedback(ctx, {"error": ""})
    arm = m.prompt_variants("coder", "code")[0]
    assert arm["uses"] == 1 and arm["wins"] == 1
    assert ctx.meta["_prompt_variant_id"] == 0  # consumed


def test_pool_underfilled_mutates_via_local_router(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    m.store_role_prompt("coder", "code", "champion mission " * 40,
                        reason="cold")
    from agentics.llm import local_router

    monkeypatch.setattr(local_router, "allowed_for",
                        lambda s, p: p == "prompt_evolve")

    class _Resp:
        ok = True
        answer = ("Improved mission: verify twice before claiming, "
                  "prefer free tools for exploratory search, keep protocols "
                  "deterministic and standard-library only, log every "
                  "failure into the personal lesson ledger, never re-assert "
                  "a refuted claim, and always attach the machine criterion "
                  "to the plan before writing code.")

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        assert purpose == "prompt_evolve"
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    ctx = _ctx(m, settings=SimpleNamespace(
        prompt_evolution_enabled=True, prompt_explore_p=0.0,
        prompt_variant_pool=3, local_llm_enabled=True))
    mission, vid = asyncio.run(pevo.maybe_variant(ctx, "coder", "code",
                                                  "champion mission " * 40))
    assert vid == 0 and mission.startswith("champion")  # explore off
    assert len(m.prompt_variants("coder", "code")) == 1  # mutation landed


def test_local_down_serves_champion(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    async def dead_ask(*a, **k):
        return None                      # local runtime absent

    monkeypatch.setattr(local_router, "ask", dead_ask)
    ctx = _ctx(m, settings=SimpleNamespace(
        prompt_evolution_enabled=True, prompt_explore_p=0.0,
        prompt_variant_pool=3, local_llm_enabled=True))
    mission, vid = asyncio.run(pevo.maybe_variant(ctx, "coder", "code",
                                                  "champion"))
    assert (mission, vid) == ("champion", 0)
    assert m.prompt_variants("coder", "code") == []


async def _never():
    raise AssertionError("not awaited")


def test_stats_aggregate(tmp_path):
    m = _mem(tmp_path)
    vid = m.store_prompt_variant("coder", "code", "v " * 100)
    m.record_prompt_variant_outcome(vid, True)
    s = m.prompt_variant_stats()
    assert s["variants"] == 1 and s["uses"] == 1 and s["wins"] == 1
