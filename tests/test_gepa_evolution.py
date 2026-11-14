"""GEPA reflective prompt evolution (roadmap #1) — hermetic unit tests.

The wave-2 DSPy-lite pool stays the selection machinery; these tests pin the
NEW half: pareto parenting, deterministic crossover, failure-lesson
injection, pool insertion with origin='gepa', and honest empty-lab behavior.
No LLM anywhere: the local router is off in these tests, so every batch is
produced by the deterministic generator.
"""
import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.evolution import gepa  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "gepa.db")


def _settings_like():
    class S:
        gepa_enabled = True
        local_llm_enabled = False          # router off -> deterministic path
        prompt_variant_pool = 3
    return S()


# ------------------------------------------------------------- pareto front
def test_pareto_front_drops_dominated():
    items = [
        {"name": "fast-bad", "score": 0.2, "cost": 10},   # cheapest: NON-dominated
        {"name": "slow-good", "score": 0.9, "cost": 100},  # best score: NON-dominated
        {"name": "sweet", "score": 0.8, "cost": 20},       # NON-dominated
        {"name": "dominated", "score": 0.8, "cost": 30},   # sweet dominates
    ]
    front = gepa.pareto_front(items)
    names = {i["name"] for i in front}
    assert names == {"fast-bad", "slow-good", "sweet"}


def test_pareto_front_tie_keeps_first_only():
    items = [{"n": "a", "score": 0.5, "cost": 10},
             {"n": "b", "score": 0.5, "cost": 10}]
    front = gepa.pareto_front(items)
    assert [i["n"] for i in front] == ["a"]


def test_pareto_front_empty():
    assert gepa.pareto_front([]) == []


# ---------------------------------------------------------------- crossover
def test_crossover_recombines_without_inventing():
    a = ("Scan the issue deeply before writing code. "
         "Always run the protocol twice. Keep citations verbatim.")
    b = ("Write the protocol before judging results. "
         "Never trust an unmeasured claim. Report failures honestly.")
    out = gepa.crossover(a, b, __import__("random").Random(7))
    parents = {s.rstrip(".") for s in gepa._sentences(a)} \
        | {s.rstrip(".") for s in gepa._sentences(b)}
    outs = [s.rstrip(".").strip() for s in out.split(". ") if s.strip()]
    assert outs and all(s in parents for s in outs)
    # both parents contributed at least one sentence
    sa = {s.rstrip(".") for s in gepa._sentences(a)}
    sb = {s.rstrip(".") for s in gepa._sentences(b)}
    assert any(s in sa for s in outs) and any(s in sb for s in outs)


def test_crossover_deterministic_same_seed():
    a = "Alpha sentence one here. Beta sentence two here."
    b = "Gamma sentence three now. Delta sentence four now."
    r1 = gepa.crossover(a, b, __import__("random").Random(42))
    r2 = gepa.crossover(a, b, __import__("random").Random(42))
    assert r1 == r2


def test_crossover_degrades_to_other_parent():
    assert gepa.crossover("", "only parent text stands here alone", None) \
        == "only parent text stands here alone"
    assert gepa.crossover("one full parent sentence here", "", None) \
        == "one full parent sentence here"


# ------------------------------------------------------------- reflection
def test_collect_reflection_reads_real_ledgers(mem):
    mem.store_role_prompt("coder", "code", mission="Be a careful coder.")
    vid = mem.store_prompt_variant("coder", "code", "Variant mission here.",
                                   origin="mutate")
    mem.record_prompt_variant_outcome(vid, True)
    mem.record_prompt_variant_outcome(vid, False)
    mem.add_trace("a01", "coder", 1, "failed", category="code",
                  failure="protocol exited non-zero")
    r = gepa.collect_reflection(mem, "coder", "code")
    assert r["champion"] == "Be a careful coder."
    assert len(r["arms"]) == 1 and r["arms"][0]["uses"] == 2
    assert abs(r["arms"][0]["score"] - 0.5) < 1e-9
    assert any("protocol exited non-zero" in ln for ln in r["failures"])


def test_offline_candidates_inject_failure_lessons(mem):
    mem.store_role_prompt("coder", "code", mission="Champion mission. Second sentence stands.")
    for _ in range(3):
        mem.add_trace("a01", "coder", 1, "failed", category="code",
                      failure="timeout on verify step")
    r = gepa.collect_reflection(mem, "coder", "code")
    texts = gepa.offline_candidates(r, n=2, seed=1)
    assert len(texts) == 2
    assert all("Avoid repeating this failure" in t for t in texts)


def test_offline_candidates_empty_lab_is_honest():
    assert gepa.offline_candidates({"champion": "", "arms": [],
                                    "failures": []}, n=3) == []


# ------------------------------------------------------- full evolve batch
def test_reflective_and_evolve_insert_gepa_arms(mem):
    mem.store_role_prompt("coder", "code", mission="Be careful. Run twice.")
    mem.add_trace("a01", "coder", 1, "failed", category="code",
                  failure="skipped the protocol")
    s = _settings_like()
    texts = asyncio.run(gepa.reflective_candidates(mem, s, "coder", "code",
                                                   n=3, seed=5))
    assert texts, "deterministic generator must produce candidates here"
    ids = gepa.evolve(mem, s, "coder", "code", texts)
    assert len(ids) == len(texts)
    pool = mem.prompt_variants("coder", "code")
    assert all(v["origin"] == "gepa" for v in pool)
    # champion untouched — GEPA only feeds the challenger pool
    assert mem.get_role_prompt("coder", "code")["mission"] == "Be careful. Run twice."


def test_pool_cap_still_applies_to_gepa_batch(mem):
    mem.store_role_prompt("coder", "code", mission="Champion text lives here.")
    s = _settings_like()
    texts = [f"Candidate mission number {i} with plenty of words to be long."
             for i in range(5)]
    ids = gepa.evolve(mem, s, "coder", "code", texts)
    assert len(ids) == 5
    armed = mem.prompt_variants("coder", "code")
    assert len(armed) == 3                    # pool_size cap retired the rest


def test_gepa_disabled_returns_nothing(mem):
    mem.store_role_prompt("coder", "code", mission="Champion text lives here.")

    class S:
        gepa_enabled = False
    texts = asyncio.run(gepa.reflective_candidates(mem, S(), "coder", "code"))
    assert texts == []


# ------------------------------------------------------------------- CLI
def test_cli_gepa_evolve_inserts(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    mem = SharedMemory(s.db_path)
    mem.store_role_prompt("coder", "code", mission="Be careful. Run twice.")
    mem.add_trace("a01", "coder", 1, "failed", category="code",
                  failure="skipped the protocol")
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["gepa-evolve", "coder", "code"])
    assert result.exit_code == 0, result.output
    assert "Đã chèn" in result.output
    mem2 = SharedMemory(s.db_path)
    pool = mem2.prompt_variants("coder", "code")
    mem2.close()
    assert pool and all(v["origin"] == "gepa" for v in pool)


def test_cli_gepa_evolve_empty_is_honest(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(
        cli.app, ["gepa-evolve", "writer", "docs", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Không có gì để lai tạo" in result.output
