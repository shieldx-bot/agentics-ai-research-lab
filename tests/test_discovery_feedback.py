"""Discovery feedback round (2026-11-14, self-improvement round 2).

Closes three gaps found by reviewing the novelty round:
1. COVERAGE — the MAIN verify path (node_verify) settled claims silently:
   no ledger row, no novelty judgment, no branching. Only the deep-dive
   path fed the loop. Now every settle feeds it (one harvest per cycle,
   supported preferred over refuted).
2. BANDIT FEEDBACK — a branch settle now credits/debits its discovery
   root: supported -> novelty +5 (discovery_confirmed), refuted ->
   branch_dead_end with the failed angle. The lab learns WHICH dig
   directions pay off instead of treating the discovery as static.
3. PINNING — DISCOVERY world notes stay pinned above the routine stream
   so breakthroughs never age out of the agents' context.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _hyp(mem, claim, status="proposed", parents=None, depth=0, branch=""):
    return mem.add_hypothesis(
        "a02", claim, {"metric": "m", "op": ">=", "target": 1},
        status=status, parents=parents, depth=depth, branch=branch)


# ------------------------------------------------------------------- memory

def test_nearest_discovery_ancestor_finds_root_at_distance(mem):
    disc = _hyp(mem, "discovery root row", status="supported")
    mem.mark_discovery(disc, 80)
    mid = _hyp(mem, "branch child row", parents=[disc], depth=1,
               branch="mechanism")
    leaf = _hyp(mem, "grandchild row", parents=[mid], depth=2)
    root = mem.nearest_discovery_ancestor(leaf)
    assert root is not None and root["id"] == disc
    assert mem.nearest_discovery_ancestor(mid)["id"] == disc
    # the discovery itself is its own nearest ancestor
    assert mem.nearest_discovery_ancestor(disc)["id"] == disc
    # unrelated row -> None
    plain = _hyp(mem, "unrelated plain proposal row")
    assert mem.nearest_discovery_ancestor(plain) is None


def test_discovery_reinforce_bonuses_and_caps(mem):
    disc = _hyp(mem, "discovery root to reinforce", status="supported")
    mem.mark_discovery(disc, 70)
    assert mem.discovery_reinforce(disc) == pytest.approx(75.0)
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(75.0)
    # cap at 100
    mem.mark_discovery(disc, 99)
    assert mem.discovery_reinforce(disc, 20) == pytest.approx(100.0)


def test_world_render_pins_discovery_notes_above_stream(mem):
    mem.world_add("a01", "verified", "DISCOVERY(novelty=92): the surprise "
                  "effect -> supported", source="h#1")
    for i in range(15):
        mem.world_add("a01", "verified", f"routine verified note number {i}",
                      source=f"h#{i + 10}")
    rendered = mem.world_render(limit=5)
    first_line = rendered.splitlines()[0]
    assert "DISCOVERY(novelty=92)" in first_line
    # routine stream still follows, bounded
    assert len(rendered.splitlines()) == 5


def test_world_items_kind_query_unpinned_kind_path(mem):
    mem.world_add("a01", "gap", "no data on churn tail latency", source="h#3")
    items = mem.world_items(kind="gap", limit=5)
    assert len(items) == 1 and "churn" in items[0]["text"]


# ----------------------------------------------------- branch feedback

class _S:
    research_topic = "NFV"
    diversity_jaccard_max = 0.8
    ideation_enabled = True
    idea_min_score = 60
    idea_repair_margin = 20
    idea_max_children = 2
    idea_max_attempts = 2
    world_context_notes = 12
    verify_protocol_repair_rounds = 0
    discovery_novelty_min = 70
    branch_max_depth = 4
    # legacy mode: the anchor law + double-judge gate have their own suite
    settle_require_anchor = False
    discovery_second_judge = False


def _ctx(mem, tmp_path, llm):
    from agentics.agents.roles import Role

    class FakeWS:
        def __init__(self, root: Path) -> None:
            import subprocess

            self.path = root
            self._subprocess = subprocess

        def sync_main(self) -> None:
            self.path.mkdir(parents=True, exist_ok=True)

        def write_file(self, rel: str, content: str) -> None:
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        def run_command(self, cmd, timeout=120):
            script = self.path / cmd[1]
            proc = self._subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True,
                timeout=timeout, cwd=self.path, check=False)
            return proc.returncode, proc.stdout + proc.stderr

    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, memory=mem, llm=llm,
        workspace=FakeWS(tmp_path / "ws"),
        cycle=1, settings=_S(),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"_system_prompt": "sys"},
    )


def _events(mem, n=15):
    return [e["event"] for e in mem.recent_journal(limit=n, agent="a02")]


def test_branch_feedback_confirms_and_dead_ends(mem, tmp_path):
    from agentics.graph.nodes import _branch_feedback

    disc = _hyp(mem, "discovery root for feedback", status="supported")
    mem.mark_discovery(disc, 80)
    good = _hyp(mem, "supported branch child row", parents=[disc], depth=1,
                branch="mechanism")
    dead = _hyp(mem, "refuted boundary branch child row", parents=[disc],
                depth=1, branch="boundary")
    ctx = _ctx(mem, tmp_path, None)
    asyncio.run(_branch_feedback(ctx, good, "supported"))
    # entropic credit (C3): the settle has NO ledger reward row here (the
    # unit call bypasses the settle path), so the group is empty and the
    # boost is 0 — a supported branch with no measurable reward must not
    # pump the discovery's novelty (the old unconditional +5 is gone).
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(80.0)
    assert "discovery_confirmed" in _events(mem)
    asyncio.run(_branch_feedback(ctx, dead, "refuted"))
    # refuted does NOT demote the discovery, it logs the dead angle
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(80.0)
    assert "branch_dead_end" in _events(mem)
    detail = next(e for e in mem.recent_journal(limit=10, agent="a02")
                  if e["event"] == "branch_dead_end")
    assert json.loads(detail["detail"])["angle"] == "boundary"
    # non-branch and self-settles are silent
    plain = _hyp(mem, "plain proposal without discovery lineage")
    asyncio.run(_branch_feedback(ctx, plain, "supported"))
    asyncio.run(_branch_feedback(ctx, disc, "refuted"))
    events = _events(mem)
    assert events.count("discovery_confirmed") == 1
    assert events.count("branch_dead_end") == 1


# --------------------------------------- node_verify full-loop coverage

PROTOCOL = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'effect_size': 0.3}))\n"
)
PROTOCOL_MS = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'cold_start_ms': 40.0}))\n"
)


def _verify_reply(n=1):
    topics = [
        "the method keeps effect above target on synthetic bursty load",
        "the cache prewarm cuts the cold-start latency below threshold",
    ]
    metrics = ["effect_size", "cold_start_ms"]
    ops = [">=", "<="]
    targets = [0.2, 50.0]
    protocols = [PROTOCOL, PROTOCOL_MS]
    return json.dumps({"hypotheses": [
        {"claim": f"verify claim {i}: {topics[i % len(topics)]}",
         "metric": metrics[i % len(metrics)], "op": ops[i % len(ops)],
         "target": targets[i % len(targets)],
         "protocol": protocols[i % len(protocols)]} for i in range(n)]})


def _discovery_reply():
    return json.dumps({
        "novelty": 92, "is_discovery": True,
        "reason": "unexplained effect the world model cannot account for",
        "hypotheses": [{
            "claim": "the verify effect is caused by the retry backoff "
                     "interaction rather than the scheduler",
            "metric": "effect_size", "op": ">=", "target": 0.1,
            "angle": "mechanism"}]})


class FakeLLM:
    account = SimpleNamespace(name="offline")

    def __init__(self, *answers: tuple[str, str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    async def ask(self, prompt, *, mode="concise", system=None,
                  purpose="general"):
        self.calls.append((purpose, prompt))
        for i, (marker, payload) in enumerate(self.answers):
            if marker in prompt:
                del self.answers[i]
                return SimpleNamespace(answer=payload, citations=[], ok=True)
        return SimpleNamespace(answer="", citations=[], ok=False)


def test_node_verify_settle_feeds_ledger_discovery_and_harvest(mem, tmp_path):
    """Main verify path: supported settle -> ledger row (source=verify) ->
    discovery marked (novelty 92) -> angled branch children registered.
    Two supported settles -> exactly ONE ideation harvest per cycle."""
    from agentics.graph.nodes import node_verify

    llm = FakeLLM(("Research finding to verify", _verify_reply(n=2)),
                  ("ROLE: ideator", _discovery_reply()),
                  ("ROLE: judge", json.dumps(
                      {"non_obviousness": 85, "clarity": 80, "feasibility": 70,
                       "soundness": 8, "notes": "solid"})),
                  ("ROLE: judge", json.dumps(
                      {"non_obviousness": 85, "clarity": 80, "feasibility": 70,
                       "soundness": 8, "notes": "solid"})))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {"finding_text": "a finding text"}
    out = asyncio.run(node_verify(ctx, state))
    assert len(out["verifications"]) == 2
    assert all(v["status"] == "supported" for v in out["verifications"])
    # ledger discipline now covers the verify path too
    ledger = mem.ledger_recent(limit=5)
    assert len(ledger) == 2
    assert all(row["source"] == "verify" for row in ledger)
    assert ledger[0]["status"] == "supported"
    # discovery recognized from the verify settle
    hyp_ids = [v["hypothesis_id"] for v in out["verifications"]]
    disc = mem.get_hypothesis(hyp_ids[0])
    assert disc["discovery"] is True
    assert disc["novelty"] == pytest.approx(92.0)
    # ONE ideation HARVEST per cycle (not per settle) — the judge calls for
    # the gated children share the purpose but carry the "ROLE: judge" prompt
    ideation_calls = [p for purpose, p in llm.calls
                      if purpose == "ideation" and "ROLE: ideator" in p]
    assert len(ideation_calls) == 1
    kids = [r for r in mem.list_hypotheses(limit=8)
            if r["parents"] == [hyp_ids[0]]]
    assert len(kids) == 1
    assert kids[0]["depth"] == 1 and kids[0]["branch"] == "mechanism"
    events = _events(mem)
    assert "discovery_detected" in events and "hypothesis_verified" in events


def test_node_verify_inconclusive_settle_stays_silent(mem, tmp_path):
    """A broken protocol (no AGENTICS_RESULT) settles inconclusive — no
    ledger keep row beyond the discard row, no harvest, no discovery."""
    from agentics.graph.nodes import node_verify

    broken = "print('nothing useful')\n"
    reply = json.dumps({"hypotheses": [{
        "claim": "an inconclusive verify claim about load shedding",
        "metric": "effect_size", "op": ">=", "target": 0.2,
        "protocol": broken}]})
    llm = FakeLLM(("Research finding to verify", reply))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {"finding_text": "a finding text"}
    out = asyncio.run(node_verify(ctx, state))
    assert out["verifications"][0]["status"] == "inconclusive"
    assert [p for purpose, p in llm.calls if purpose == "ideation"] == []
    assert "discovery_detected" not in _events(mem)
    ledger = mem.ledger_recent(limit=5)
    assert all(row["status"] != "supported" for row in ledger)
