"""P24bis — node_replicate + routing: the live cycle replicates claims.

Chain built for real: the author's node_verify settles a supported claim
(stored protocol, confirmed evidence), then a REVIEWER agent's
node_replicate independently re-runs the claim in its OWN workspace and the
swarm's common knowledge is flagged `replicated`. Fake LLM only where a
node would ask one — the replicator itself needs no model at all.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory

PROTOCOL = (
    "import json\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('simulated eval done')\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
WEAKER_PROTOCOL = PROTOCOL.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")


class AuthorLLM:
    """Answers node_verify's extraction prompt with one verifiable claim."""

    account = SimpleNamespace(name="offline")

    async def ask(self, query, *, mode="concise", system=None, purpose="general"):
        if purpose != "verify":
            return SimpleNamespace(answer="", citations=[], ok=False)
        payload = {"hypotheses": [
            {"claim": "the method reaches >=0.9 precision on the toy set",
             "metric": "precision", "op": ">=", "target": 0.9,
             "protocol": PROTOCOL},
        ]}
        return SimpleNamespace(answer=json.dumps(payload), citations=[], ok=True)


class RunWS:
    """Plain-directory workspace with the runner contract (no git needed)."""

    def __init__(self, root: Path) -> None:
        self.path = root

    def sync_main(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        script = self.path / cmd[1]
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=timeout, cwd=self.path,
                              check=False)
        return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _ctx(mem: SharedMemory, tmp_path: Path, role: Role,
         agent_id: str, *, llm=None, replication_limit: int = 1) -> SimpleNamespace:
    ws_root = tmp_path / f"ws-{agent_id}"
    ws_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        agent_id=agent_id, role=role, memory=mem, llm=llm,
        workspace=RunWS(ws_root), cycle=1,
        settings=SimpleNamespace(research_topic="NFV",
                                 replication_limit=replication_limit),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
    )


def _state(**over) -> dict:
    base = {"done": False, "finding_id": 5, "issue_number": None,
            "finding_text": "Finding: RF reaches ~0.9 precision..."}
    base.update(over)
    return base


def _settled_claim(mem: SharedMemory, tmp_path: Path) -> dict:
    """The author's real verify node run: claim -> protocol -> supported."""
    author = _ctx(mem, tmp_path, Role.RESEARCHER, "researcher-1",
                  llm=AuthorLLM())
    out = asyncio.run(nodes.node_verify(author, _state()))
    assert "supported:1" in out["summary"]
    return mem.list_hypotheses()[0]


def test_reviewer_replicates_in_its_own_workspace(mem, tmp_path):
    hyp = _settled_claim(mem, tmp_path)
    assert mem.verified_knowledge()[0]["replicated"] is False
    reviewer = _ctx(mem, tmp_path, Role.REVIEWER, "reviewer-1")
    out = asyncio.run(nodes.node_replicate(reviewer, _state()))

    assert out["done"] and "confirmed:1" in out["summary"]
    assert out.get("replication_verdict") == "confirmed:1"
    row = mem.get_hypothesis(hyp["id"])
    assert row["replication"] == "confirmed"       # canonical ledger
    vk = mem.verified_knowledge()[0]
    assert vk["replicated"] is True                # common knowledge upgraded
    detail = mem.replication_for_evidence(hyp["evidence_id"])
    assert detail["replicator"] == "reviewer-1"
    # rebuilt INSIDE the reviewer's workspace — never the author's
    assert (tmp_path / "ws-reviewer-1" / "experiments" /
            f"replicate_h{hyp['id']}.py").exists()
    assert not (tmp_path / "ws-researcher-1" / "experiments" /
                f"replicate_h{hyp['id']}.py").exists()
    events = [j["event"] for j in mem.recent_journal()]
    assert "replication_confirmed" in events


def test_refuted_replication_via_node_overrides_the_author(mem, tmp_path):
    hyp = _settled_claim(mem, tmp_path)
    reviewer = _ctx(mem, tmp_path, Role.REVIEWER, "reviewer-1")
    # the reviewer re-runs a WEAKER simulation — a real disagreement between
    # peers, fed straight to the engine (the node path is tested above)
    cand = dict(mem.hypotheses_needing_replication("reviewer-1")[0],
                protocol_code=WEAKER_PROTOCOL)
    from agentics.verify.replication import replicate_candidate
    report = asyncio.run(asyncio.to_thread(
        replicate_candidate, mem, reviewer.workspace, cand, "reviewer-1"))
    assert report["verdict"] == "refuted"
    row = mem.get_hypothesis(hyp["id"])
    assert row["status"] == "refuted" and row["replication"] == "refuted"
    ev_row = [r for r in mem.latest_evidence(limit=5)
              if r["id"] == hyp["evidence_id"]][0]
    assert ev_row["verdict"] == "refuted"
    assert mem.verified_knowledge()[0]["status"] == "refuted"


def test_empty_queue_and_disabled_knob_are_clean_noops(mem, tmp_path):
    reviewer = _ctx(mem, tmp_path, Role.REVIEWER, "reviewer-1")
    out = asyncio.run(nodes.node_replicate(reviewer, _state()))
    assert out["done"] and "no claim awaiting" in out["summary"]
    off = _ctx(mem, tmp_path, Role.REVIEWER, "reviewer-2",
               replication_limit=0)
    out = asyncio.run(nodes.node_replicate(off, _state()))
    assert out["done"] and "disabled" in out["summary"]


def test_router_sends_reviewer_to_replicate_first(monkeypatch):
    """P24bis routing: pending replication outranks ordinary review (it is
    cheap — no LLM — and common knowledge resting on ONE run is the most
    urgent problem); with an empty queue the reviewer falls through."""
    pytest.importorskip("langgraph")
    import agentics.graph.research_graph as rg

    calls: list[str] = []

    def make(name):
        # mirror the real graph: only the LLM nodes are registered async
        if name in {"research", "plan", "implement", "verify", "review",
                    "defend", "replicate"}:
            async def async_node(ctx, state):
                calls.append(name)
                state["done"] = True
                return state
            return async_node

        def node(ctx, state):
            calls.append(name)
            state["done"] = True
            return state
        return node

    for n in ("start", "scan", "claim", "research", "plan", "implement",
              "open_pr", "review", "replicate", "defend", "verify",
              "integrate", "reflect"):
        monkeypatch.setattr(rg, f"node_{n}", make(n))

    def _run(pending, defense):
        calls.clear()
        monkeypatch.setattr(rg, "pending_replication", lambda ctx: pending)
        monkeypatch.setattr(rg, "pending_defense", lambda ctx: defense)
        asyncio.run(rg.build_research_graph(
            SimpleNamespace(role=Role.REVIEWER)).compile().ainvoke({}))
        return calls

    # a settled unreplicated claim exists -> replicate first
    assert _run(pending=[{"id": 1}], defense=None) == \
        ["start", "replicate", "reflect"]
    # nothing to replicate -> ordinary review
    assert _run(pending=[], defense=None) == ["start", "review", "reflect"]
    # defense ALWAYS outranks replication (P19 priority preserved); defend
    # has no unconditional reflect edge — it routes internally
    assert _run(pending=[{"id": 1}], defense=(SimpleNamespace(),
                                               SimpleNamespace())) == \
        ["start", "defend"]
    # a crashed routing read must never kill the cycle -> fall to review
    def boom(ctx):
        raise RuntimeError("db hiccup")
    monkeypatch.setattr(rg, "pending_replication", boom)
    monkeypatch.setattr(rg, "pending_defense", boom)
    calls.clear()
    asyncio.run(rg.build_research_graph(
        SimpleNamespace(role=Role.REVIEWER)).compile().ainvoke({}))
    assert calls == ["start", "review", "reflect"]
