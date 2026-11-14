"""Wave-2 #5 — reflection tree.

Contract under test:
* insights are stored ONLY with valid provenance (real parent note ids);
* a fabricated parent list is dropped wholesale — never stored;
* reflection depth is parent-level + 1, capped at 3;
* cadence gate: one pass every reflect_every_cycles cycles, none between;
* everything runs on the LOCAL router — the cookie pool is untouched.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.graph import reflection as refl  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _mem(tmp_path):
    return SharedMemory(tmp_path / "refl.db")


def _ctx(memory, cycle=6, **settings_kw):
    base = dict(agent_id="a07", memory=memory, cycle=cycle,
                settings=SimpleNamespace(
                    reflect_every_cycles=6, reflect_min_notes=5,
                    local_llm_enabled=True), journal=lambda *a, **k: None)
    base.update(settings_kw)
    return SimpleNamespace(**base)


def _seed_notes(m, n=6):
    ids = []
    for i in range(n):
        kind = "verified" if i % 2 == 0 else "refuted"
        ids.append(m.world_add("a01", kind, f"note text {i} about metric",
                               source=f"evidence#{i}"))
    return ids


# ------------------------------------------------------- world model depth
def test_world_notes_carry_level_and_parents(tmp_path):
    m = _mem(tmp_path)
    pid = m.world_add("a01", "verified", "raw note", source="evidence#1")
    iid = m.world_add("a02", "insight", "synthesized pattern",
                      source="reflect#6", level=1, parents=[pid])
    items = m.world_items()
    by_id = {i["id"]: i for i in items}
    assert by_id[pid]["level"] == 0 and by_id[pid]["parents"] == []
    assert by_id[iid]["level"] == 1 and by_id[iid]["parents"] == [pid]


def test_world_render_marks_insight_depth(tmp_path):
    m = _mem(tmp_path)
    m.world_add("a01", "verified", "raw note", source="evidence#1")
    m.world_add("a01", "insight", "level one insight", source="reflect#6",
                level=1, parents=[1])
    rendered = m.world_render(limit=5)
    assert "[INSIGHT-L1]" in rendered and "[VERIFIED]" in rendered


# ------------------------------------------------------------ provenance
def test_parents_valid_gate(tmp_path):
    known = {1, 2, 3}
    assert refl._parents_valid([1, 2], known) == [1, 2]
    assert refl._parents_valid([1, 1, 2], known) == [1, 2]   # dedup
    assert refl._parents_valid([999], known) == []           # fabricated
    assert refl._parents_valid("nope", known) == []
    assert refl._parents_valid([None, "x", 2], known) == [2]


# ------------------------------------------------------------ reflect_once
def test_reflection_stores_valid_insight_only(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    ids = _seed_notes(m, 6)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    good = ids[0]
    class _Resp:
        ok = True
        answer = (
            '```json\n{"insights": ['
            '{"text": "protocols using metric X keep failing on '
            'determinism", "parents": [' + str(good) + ']}, '
            '{"text": "invented claim with no source", '
            '"parents": [424242]}]}\n```')

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        assert purpose == "reflect"
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    stored = asyncio.run(refl.reflect_once(_ctx(m)))
    assert stored == 1                       # fabricated parent dropped
    insights = [i for i in m.world_items() if i["kind"] == "insight"]
    assert len(insights) == 1
    assert insights[0]["level"] == 1 and insights[0]["parents"] == [good]


def test_reflection_level_2_from_level_1_parents(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    ids = _seed_notes(m, 5)
    l1 = m.world_add("a01", "insight", "level one insight", level=1,
                     parents=[ids[0]])
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    class _Resp:
        ok = True
        answer = (f'{{"insights": [{{"text": "level two synthesis", '
                  f'"parents": [{l1}]}}]}}')

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    assert asyncio.run(refl.reflect_once(_ctx(m))) == 1
    ins = [i for i in m.world_items() if i["kind"] == "insight"
           and i["level"] == 2]
    assert len(ins) == 1 and ins[0]["parents"] == [l1]


def test_reflection_needs_enough_notes(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    _seed_notes(m, 3)                        # < reflect_min_notes
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    async def boom(prompt, settings=None, purpose="", system=""):
        raise AssertionError("router must not be called")

    monkeypatch.setattr(local_router, "ask", boom)
    assert asyncio.run(refl.reflect_once(_ctx(m))) == 0


def test_reflection_disabled_returns_zero(tmp_path):
    m = _mem(tmp_path)
    _seed_notes(m, 6)
    ctx = _ctx(m, local_llm_enabled=False)
    assert asyncio.run(refl.reflect_once(ctx)) == 0


def test_garbage_answer_stores_nothing(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    _seed_notes(m, 6)
    from agentics.llm import local_router
    monkeypatch.setattr(local_router, "allowed_for", lambda s, p: True)

    class _Resp:
        ok = True
        answer = "I think the lab is doing great, no JSON here."

    async def fake_ask(prompt, settings=None, purpose="", system=""):
        return _Resp()

    monkeypatch.setattr(local_router, "ask", fake_ask)
    assert asyncio.run(refl.reflect_once(_ctx(m))) == 0
    assert [i for i in m.world_items() if i["kind"] == "insight"] == []


# --------------------------------------------------------------- cadence
def test_maybe_reflect_cadence_gate(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    calls = []

    async def fake_once(ctx):
        calls.append(ctx.cycle)

    monkeypatch.setattr(refl, "reflect_once", fake_once)
    refl.maybe_reflect(_ctx(m, cycle=3))     # 3 % 6 != 0 -> skip
    assert calls == []
    refl.maybe_reflect(_ctx(m, cycle=12))    # 12 % 6 == 0 -> run
    assert calls == [12]


def test_maybe_reflect_disabled_via_zero(tmp_path):
    m = _mem(tmp_path)
    ctx = _ctx(m, cycle=6, reflect_every_cycles=0)
    refl.maybe_reflect(ctx)                  # must be a no-op, never raise
