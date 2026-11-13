"""Discovery detection + branch deep-dive (2026-11-14 novelty round).

The user's ask: other research agents don't just rotate ideas — when an
experiment PAYS OFF, the agent must RECOGNIZE that it found something new
and BRANCH OUT to dig into that finding. Mechanisms under test:

- autoresearch-style quantitative novelty judgment: the ideator grades the
  settled result against the world model (novelty 0-100, is_discovery).
- Registry DISCOVERY flag (mark_discovery) + branch lineage columns
  (depth, branch angle).
- Angled branches: mechanism / generalize / boundary / amplify — children
  attack the discovery from DIFFERENT angles, not re-wordings.
- Discovery frontier: the deep-dive backlog prioritizes the discovery's
  subtree (BFS, shallowest first) over generic proposals, with a
  branch_max_depth cap so one finding cannot monopolize the lab.
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

from agentics.research import ideation as idea  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


# ------------------------------------------------------------ pure parsers

def test_parse_discovery_full_contract():
    raw = json.dumps({
        "novelty": 92, "is_discovery": True,
        "reason": "result contradicts a verified world-model note",
        "hypotheses": [
            {"claim": "the surprise effect persists under a heavier tail "
                      "workload than the original benchmark used",
             "metric": "p99_ms", "op": "<", "target": 200,
             "angle": "generalize"},
            {"claim": "the effect vanishes once the cache is fully warm",
             "metric": "speedup", "op": ">=", "target": 1.5,
             "angle": "boundary", "tolerance": 0.1},
        ]})
    novelty, is_disc, reason, children = idea.parse_discovery(raw, 2)
    assert novelty == pytest.approx(92.0) and is_disc
    assert "contradicts" in reason
    assert len(children) == 2
    assert children[0]["angle"] == "generalize"
    assert children[1]["criterion"]["tolerance"] == 0.1


def test_parse_discovery_missing_novelty_is_not_a_discovery():
    # replies predating the novelty contract (or omitting it) must degrade
    # to the old follow-up behavior, never guess
    old = json.dumps({"hypotheses": [
        {"claim": "a plain follow-up claim about autoscaling thresholds",
         "metric": "m", "op": ">=", "target": 1}]})
    novelty, is_disc, reason, children = idea.parse_discovery(old, 2)
    assert novelty is None and not is_disc and reason == ""
    assert len(children) == 1
    assert "angle" not in children[0]
    assert idea.parse_discovery("prose only", 2) == (None, False, "", [])


def test_parse_discovery_rejects_novelty_out_of_range_and_type():
    hi = json.dumps({"novelty": 250, "is_discovery": True, "hypotheses": []})
    assert idea.parse_discovery(hi, 2)[0] == 100.0
    bad = json.dumps({"novelty": "very high", "is_discovery": True})
    assert idea.parse_discovery(bad, 2)[:3] == (None, False, "")


def test_child_angle_only_valid_angles_survive():
    ok = idea._child_from_dict({
        "claim": "an angled branch claim about the discovery mechanism",
        "metric": "m", "op": ">=", "target": 1, "angle": "MECHANISM"})
    assert ok["angle"] == "mechanism"
    dropped = idea._child_from_dict({
        "claim": "an angled branch claim about the discovery mechanism",
        "metric": "m", "op": ">=", "target": 1, "angle": "vibes"})
    assert "angle" not in dropped


# ------------------------------------------------------------------- memory

@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _hyp(mem, claim, status="proposed", parents=None, depth=0, branch=""):
    return mem.add_hypothesis(
        "a02", claim, {"metric": "m", "op": ">=", "target": 1},
        status=status, parents=parents, depth=depth, branch=branch)


def test_mark_discovery_sets_flag_and_novelty(mem):
    hid = _hyp(mem, "a settled claim that surprised the world model",
               status="supported")
    assert mem.mark_discovery(hid, 88) is True
    row = mem.get_hypothesis(hid)
    assert row["discovery"] is True
    assert row["novelty"] == pytest.approx(88.0)
    assert row["depth"] == 0 and row["branch"] == ""


def test_discovery_frontier_lineage_bfs_and_depth_cap(mem):
    disc = _hyp(mem, "discovery root claim", status="supported")
    mem.mark_discovery(disc, 90)
    c1 = _hyp(mem, "branch child one of the discovery", parents=[disc],
              depth=1, branch="mechanism")
    c2 = _hyp(mem, "branch child two of the discovery", parents=[disc],
              depth=1, branch="boundary")
    g1 = _hyp(mem, "grandchild of the discovery branch", parents=[c1],
              depth=2, branch="amplify")
    plain = _hyp(mem, "a plain proposal unrelated to any discovery")
    deep = _hyp(mem, "a too-deep descendant past the cap", parents=[g1],
                depth=5, branch="mechanism")

    got = mem.discovery_frontier(limit=10, max_depth=4)
    ids = [h["id"] for h in got]
    # shallowest-first BFS; plain and past-cap rows excluded
    assert ids == [c1, c2, g1]
    assert plain not in ids and deep not in ids
    assert got[0]["branch"] == "mechanism"
    # depth cap 1: only the two depth-1 children
    assert [h["id"] for h in mem.discovery_frontier(limit=10, max_depth=1)] \
        == [c1, c2]


def test_discovery_frontier_survives_parent_cycles(mem):
    disc = _hyp(mem, "discovery root for cycle test", status="supported")
    mem.mark_discovery(disc, 75)
    a = _hyp(mem, "cycle member alpha branch claim", parents=[disc], depth=1)
    b = _hyp(mem, "cycle member beta branch claim", parents=[a], depth=2)
    # inject a parent cycle a -> b -> a while KEEPING a's real lineage to
    # the discovery root (malformed lineage must not hang or drop rows)
    with mem._lock, mem._conn:
        mem._conn.execute(
            "UPDATE hypotheses SET parents=? WHERE id=?",
            (json.dumps([disc, b]), a))
    got = mem.discovery_frontier(limit=10, max_depth=4)
    assert [h["id"] for h in got] == [a, b]


# ------------------------------------------------------- harvest branching

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


GOOD_JUDGE = json.dumps({"non_obviousness": 85, "clarity": 80,
                         "feasibility": 70, "soundness": 8,
                         "notes": "solid mechanism, clean test path"})
LOW_JUDGE = json.dumps({"non_obviousness": 30, "clarity": 40,
                        "feasibility": 60, "soundness": 4,
                        "notes": "routine ablation of known work"})


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


def _ctx(mem, tmp_path, llm):
    from agentics.agents.roles import Role

    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, memory=mem, llm=llm,
        workspace=FakeWS(tmp_path / "ws"),
        cycle=1, settings=_S(),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"_system_prompt": "sys"},
    )


def _angled_children_reply(novelty=92, is_disc=True):
    return json.dumps({
        "novelty": novelty, "is_discovery": is_disc,
        "reason": "an unexplained effect the world model cannot account for",
        "hypotheses": [
            {"claim": "the discovery effect is caused by the retry backoff "
                      "interaction rather than the scheduler",
             "metric": "effect_size", "op": ">=", "target": 0.2,
             "angle": "mechanism"},
            {"claim": "the discovery effect transfers to a workload with "
                      "three times the arrival rate",
             "metric": "effect_size", "op": ">=", "target": 0.1,
             "angle": "generalize"},
        ]})


def test_harvest_marks_discovery_and_registers_angled_branches(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim about autoscaling policies",
                  status="supported")
    llm = FakeLLM(("ROLE: ideator", _angled_children_reply()),
                  ("ROLE: judge", GOOD_JUDGE),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    n = asyncio.run(_harvest_insights(ctx, parent, "parent claim",
                                      "supported", "speedup=2.4"))
    assert n == 2
    # the agent RECOGNIZED the discovery
    row = mem.get_hypothesis(parent)
    assert row["discovery"] is True
    assert row["novelty"] == pytest.approx(92.0)
    events = {e["event"] for e in mem.recent_journal(limit=12, agent="a02")}
    assert "discovery_detected" in events
    assert "insight_harvest" in events
    # the world model records the discovery for every other agent
    world = mem.world_render(limit=12)
    assert "DISCOVERY(novelty=92)" in world
    # branches: depth+1 + angle stored on each child
    kids = [r for r in mem.list_hypotheses(limit=6) if r["parents"] == [parent]]
    assert len(kids) == 2
    assert all(k["depth"] == 1 for k in kids)
    assert {k["branch"] for k in kids} == {"mechanism", "generalize"}
    # frontier now returns them, shallowest first
    got = mem.discovery_frontier(limit=5, max_depth=4)
    assert [h["id"] for h in got] == sorted(k["id"] for k in kids)


def test_harvest_low_novelty_keeps_old_followup_behavior(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled routine confirmation claim", status="supported")
    reply = json.dumps({
        "novelty": 45, "is_discovery": False,
        "reason": "expected extension of a known result",
        "hypotheses": [{
            "claim": "a routine follow-up about threshold autoscaling",
            "metric": "m", "op": ">=", "target": 1}]})
    llm = FakeLLM(("ROLE: ideator", reply), ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    n = asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert n == 1
    row = mem.get_hypothesis(parent)
    assert row["discovery"] is False           # NOT marked
    kid = [r for r in mem.list_hypotheses(limit=5) if r["parents"] == [parent]][0]
    assert kid["branch"] == ""                 # not an angled branch
    events = [e["event"] for e in mem.recent_journal(limit=8, agent="a02")]
    assert "discovery_detected" not in events


def test_harvest_old_contract_reply_without_novelty(mem, tmp_path):
    """Replies predating the novelty contract keep the pre-discovery loop."""
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim for old contract", status="supported")
    reply = json.dumps({"hypotheses": [{
        "claim": "an old-style follow-up claim about queue depth limits",
        "metric": "m", "op": ">=", "target": 1}]})
    llm = FakeLLM(("ROLE: ideator", reply), ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    n = asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert n == 1
    assert mem.get_hypothesis(parent)["discovery"] is False


def test_harvest_branch_depth_cap_stops_spawning(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled deep branch claim", status="supported",
                  depth=4)
    llm = FakeLLM(("ROLE: ideator", _angled_children_reply()))
    ctx = _ctx(mem, tmp_path, llm)
    n = asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert n == 0
    assert llm.calls and llm.calls[0][0] == "ideation"  # judged first, then capped
    events = [e["event"] for e in mem.recent_journal(limit=8, agent="a02")]
    assert "branch_depth_cap" in events
    assert not [r for r in mem.list_hypotheses(limit=5)
                if r["parents"] == [parent]]


# ------------------------------------------------- deep-dive frontier boost

PROTOCOL = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'effect_size': 0.3}))\n"
)


def test_deep_dive_prefers_discovery_branch_over_older_plain(mem, tmp_path):
    """The backlog digs into the discovery's branch even though an older
    plain proposal sits ahead of it in id order (frontier priority)."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    disc = _hyp(mem, "discovery root: cache hit rate doubles under churn",
                status="supported")
    mem.mark_discovery(disc, 91)
    older_plain = _hyp(mem, "an older plain proposal about log rotation")
    branch_kid = _hyp(mem, "the discovery effect is caused by retry backoff "
                           "interaction rather than the scheduler",
                      parents=[disc], depth=1, branch="mechanism")
    protocol_reply = json.dumps({"hypotheses": [{
        "claim": "the discovery effect is caused by retry backoff "
                 "interaction rather than the scheduler",
        "metric": "effect_size", "op": ">=", "target": 0.2,
        "protocol": PROTOCOL}]})
    llm = FakeLLM(("pre-registered by the swarm", protocol_reply),
                  ("ROLE: ideator", json.dumps({"hypotheses": []})))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    assert asyncio.run(_test_proposed_hypothesis(ctx, state)) is True
    # the BRANCH child was picked (not the older plain row)
    assert mem.get_hypothesis(branch_kid)["status"] == "supported"
    assert mem.get_hypothesis(older_plain)["status"] == "proposed"
    events = [e["event"] for e in mem.recent_journal(limit=10, agent="a02")]
    assert "frontier_priority" in events
    fp = next(e for e in mem.recent_journal(limit=10, agent="a02")
              if e["event"] == "frontier_priority")
    assert json.loads(fp["detail"])["hyp"] == branch_kid
    assert json.loads(fp["detail"])["branch"] == "mechanism"
