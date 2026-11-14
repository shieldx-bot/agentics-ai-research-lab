"""Wave-3 #1 — GEPA-style reflective prompt regeneration (evolution/
gepa_driver.py).

Contract under test:
* healthy champions (no failure markers in the recent experience) get NO
  reflection — the plain mutation path keeps serving them;
* a failing champion gets a critique-then-rewrite arm stored through the
  ordinary store_prompt_variant machinery (origin='gepa'), journalled as
  gepa_reflection;
* the local router is the ONLY generation channel (allowed_for gate);
* flag off / unusable answers degrade to 0 (no arm, no crash).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.evolution import gepa_driver as gd  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402

_CHAMPION = ("You are a researcher. Read the evidence, propose one "
             "falsifiable claim, and record the exact files you touched "
             "so the reviewer can re-derive every step you took.")


def _mem(tmp_path):
    return SharedMemory(tmp_path / "gepa.db")


def _ctx(memory, cycle=9, **settings_kw):
    base = dict(gepa_reflection_enabled=True, prompt_variant_pool=3,
                local_llm_enabled=True,
                local_llm_purposes="kg_extract,plan,role_synthesis,"
                                   "prompt_evolve,reflect,skill_distill")
    base.update(settings_kw)
    return SimpleNamespace(agent_id="a05", memory=memory, cycle=cycle,
                           settings=SimpleNamespace(**base), meta={})


def _failing_history(memory):
    for i in range(4):
        memory.add_trace("a05", role="researcher", cycle=i,
                         outcome="failed" if i % 2 else "approved",
                         category="efficiency",
                         attempted="tried the usual route",
                         failure="protocol crashed on empty input")


# ------------------------------------------------------------------ parser
def test_parse_critique_mission():
    answer = ("CRITIQUE: The mission never says what to do on empty input.\n"
              "MISSION: " + ("Always guard empty input. " * 12))
    critique, mission = gd.parse_critique_mission(answer)
    assert "empty input" in critique
    assert mission.startswith("Always guard")
    assert gd.parse_critique_mission("") == ("", "")
    assert gd.parse_critique_mission("no markers at all") == ("", "")


def _stub_ask(answer):
    async def ask(prompt, settings=None, purpose="local", system=""):
        return SimpleNamespace(answer=answer, citations=[], ok=True)
    return ask


# -------------------------------------------------------------- reflection
def test_healthy_champion_skips_reflection(tmp_path):
    m = _mem(tmp_path)
    m.add_trace("a05", role="researcher", cycle=1, outcome="approved",
                category="efficiency", attempted="clean run")
    ctx = _ctx(m)
    assert asyncio.run(gd.regenerate(ctx, "researcher", "efficiency",
                                     _CHAMPION)) == 0


def test_failing_champion_gets_gepa_arm(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    _failing_history(m)
    ctx = _ctx(m)
    mission = ("You are a researcher. ALWAYS guard empty inputs before "
               "running a protocol, record every file you touch, and stop "
               "after the first irrecoverable parse failure instead of "
               "retrying blindly. Cite the exact evidence rows you used.")
    assert len(mission) >= 200
    monkeypatch.setattr("agentics.llm.local_router.ask",
                        _stub_ask("CRITIQUE: no empty-input guard.\n"
                                  "MISSION: " + mission))
    vid = asyncio.run(gd.regenerate(ctx, "researcher", "efficiency",
                                    _CHAMPION))
    assert vid > 0
    pool = m.prompt_variants("researcher", "efficiency")
    assert any(v["id"] == vid for v in pool)
    row = m._conn.execute(
        "SELECT origin FROM prompt_variants WHERE id=?", (vid,)).fetchone()
    assert row[0] == "gepa"
    ev = m._conn.execute(
        "SELECT detail FROM journal WHERE event='gepa_reflection' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert ev is not None


def test_flag_off_and_local_off_degrade(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    _failing_history(m)
    # flag off
    assert asyncio.run(gd.regenerate(_ctx(m, gepa_reflection_enabled=False),
                                     "researcher", "efficiency",
                                     _CHAMPION)) == 0
    # local router not allowed for prompt_evolve
    ctx = _ctx(m, local_llm_enabled=False)
    assert asyncio.run(gd.regenerate(ctx, "researcher", "efficiency",
                                     _CHAMPION)) == 0


def test_unusable_answer_stores_nothing(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    _failing_history(m)
    ctx = _ctx(m)
    monkeypatch.setattr("agentics.llm.local_router.ask",
                        _stub_ask("MISSION: too short"))
    assert asyncio.run(gd.regenerate(ctx, "researcher", "efficiency",
                                     _CHAMPION)) == 0
    assert m.prompt_variants("researcher", "efficiency") == []


def test_prompts_pool_prefers_gepa_then_falls_back(tmp_path, monkeypatch):
    """_mutate_into_pool routes through the GEPA driver first; when it
    produces an arm, the plain mutation prompt never runs."""
    from agentics.evolution import prompts as pp
    m = _mem(tmp_path)
    _failing_history(m)
    ctx = _ctx(m)
    called = {"gepa": 0, "mutate": 0}

    async def fake_gepa(ctx_, role, category, champion):
        called["gepa"] += 1
        return 0                      # driver declines (e.g. unusable)

    async def fake_local_ask(prompt, settings=None, purpose="local",
                             system=""):
        called["mutate"] += 1
        return None

    monkeypatch.setattr(gd, "regenerate", fake_gepa)
    monkeypatch.setattr(pp.local_router, "ask", fake_local_ask)
    asyncio.run(pp._mutate_into_pool(ctx, "researcher", "efficiency",
                                     _CHAMPION, 3))
    assert called["gepa"] == 1 and called["mutate"] == 1
