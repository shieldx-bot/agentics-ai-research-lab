"""P24 — independent replication through the live verify node.

A researcher cycle with NOTHING new to verify still re-runs someone else's
settled claim from its stored protocol (P23 source, P23 replay engine) in its
own workspace: one agent's verdict is not knowledge — an independent re-run
is. Fake LLM, real SharedMemory, real protocol execution; no network, no git.
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

CRITERION = {"metric": "precision", "op": ">=", "target": 0.9}

PASS_PROTOCOL = (
    "import json\n"
    "# tiny simulation: 200 predictions, ~90% correct\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
FAIL_PROTOCOL = PASS_PROTOCOL.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")
CRASH_PROTOCOL = "import sys\nprint('boom')\nsys.exit(3)\n"


class FakeLLM:
    account = SimpleNamespace(name="offline")

    async def ask(self, query, *, mode="concise", system=None, purpose="general"):
        return SimpleNamespace(answer='{"hypotheses": []}', citations=[],
                               ok=True)


class FakeWS:
    """Plain directory workspace: real protocol execution, same contract."""

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


def _ctx(mem: SharedMemory, tmp_path: Path, agent_id: str = "a02",
         llm=None) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id, role=Role.RESEARCHER, memory=mem,
        llm=llm if llm is not None else FakeLLM(),
        workspace=FakeWS(tmp_path / f"ws-{agent_id}"),
        cycle=1, settings=SimpleNamespace(research_topic="NFV"),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
    )


def _state(**over) -> dict:
    base = {"done": False, "finding_id": None, "issue_number": None,
            "finding_text": ""}
    base.update(over)
    return base


def _seed_supported(mem: SharedMemory, agent: str, claim: str,
                    protocol: str = PASS_PROTOCOL) -> int:
    """Author's own run: confirmed protocol row + settled 'supported' claim."""
    ev_id = mem.record_evidence(agent, None, "experiments/x.py", 0,
                                "confirmed", hypothesis=claim, method="protocol")
    hyp_id = mem.add_hypothesis(agent, claim, CRITERION, status="testing",
                                protocol_code=protocol)
    mem.attach_protocol_outcome(ev_id, {"precision": 0.92}, [], hyp_id)
    mem.update_hypothesis(hyp_id, "supported", evidence_id=ev_id,
                          note="measured precision=0.92 >= 0.9")
    return hyp_id


def _events(mem: SharedMemory) -> list[str]:
    return [j["event"] for j in mem.recent_journal()]


def test_independent_replication_confirms(mem, tmp_path):
    hyp = _seed_supported(mem, "a01", "the method reaches >=0.9 precision")
    ctx = _ctx(mem, tmp_path, agent_id="a02")
    out = asyncio.run(nodes.node_verify(ctx, _state()))

    assert "P24 replicated 1 claim(s) [confirmed:1]" in out["summary"]
    assert out["verifications"][0]["replication"] is True
    row = mem.get_hypothesis(hyp)
    assert row["replication"] == "confirmed" and row["replication_attempts"] == 1
    assert row["status"] == "supported" and row["evidence_id"]  # stays settled
    replays = [e for e in mem.latest_evidence() if e["method"] == "replay"]
    assert len(replays) == 1 and replays[0]["hypothesis_id"] == hyp
    events = _events(mem)
    assert "claim_reconfirmed" in events and "hypothesis_replicated" in events


def test_second_cycle_never_re_replicates(mem, tmp_path):
    _seed_supported(mem, "a01", "the method reaches >=0.9 precision")
    ctx = _ctx(mem, tmp_path, agent_id="a02")
    asyncio.run(nodes.node_verify(ctx, _state()))
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert out["verifications"] == []
    assert out["summary"] == "verification skipped (no finding or no LLM)"
    assert sum(1 for e in mem.latest_evidence()
               if e["method"] == "replay") == 1


def test_refuted_replication_closes_the_ledger(mem, tmp_path):
    """A supported claim whose protocol actually fails: the independent
    replica falsifies it and verified_knowledge self-corrects."""
    hyp = _seed_supported(mem, "a01", "the method reaches >=0.9 precision",
                          protocol=FAIL_PROTOCOL)
    ctx = _ctx(mem, tmp_path, agent_id="a02")
    out = asyncio.run(nodes.node_verify(ctx, _state()))

    assert "refuted:1" in out["summary"]
    row = mem.get_hypothesis(hyp)
    assert row["status"] == "refuted" and row["replication"] == "refuted"
    knowledge = {k["id"]: k for k in mem.verified_knowledge()}
    assert knowledge[hyp]["status"] == "refuted"
    events = _events(mem)
    assert "claim_falsified" in events
    assert "hypothesis_replication_refuted" in events


def test_unmeasured_replay_settles_inconclusive(mem, tmp_path):
    """A protocol that RUNS but measures nothing: the P23 engine re-settles
    the claim to inconclusive (claim_unstable) — it leaves verified_knowledge
    and a fresh agent re-verifies it from scratch later."""
    hyp = _seed_supported(mem, "a01", "the method reaches >=0.9 precision",
                          protocol=CRASH_PROTOCOL)   # exit 3, no marker
    ctx = _ctx(mem, tmp_path, agent_id="a02")
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert "inconclusive:1" in out["summary"]
    row = mem.get_hypothesis(hyp)
    assert row["status"] == "inconclusive" and row["replication"] == "inconclusive"
    assert "claim_unstable" in _events(mem)
    # an inconclusive claim is no longer 'supported' — never re-picked
    out2 = asyncio.run(nodes.node_verify(ctx, _state()))
    assert out2["verifications"] == []


def test_crashed_replay_stays_supported_with_retry_budget(mem, tmp_path):
    """A replay the WORKSPACE cannot even execute (runner crash): the engine
    declines (replayed=False), the claim stays supported and the ledger's
    2-attempt retry budget gives it two more chances before giving up."""
    hyp = _seed_supported(mem, "a01", "the method reaches >=0.9 precision")
    ctx = _ctx(mem, tmp_path, agent_id="a02")

    class BrokenWS(FakeWS):
        def run_command(self, cmd, timeout=120):
            raise RuntimeError("process group exploded")

    ctx.workspace = BrokenWS(tmp_path / "ws-broken")
    out1 = asyncio.run(nodes.node_verify(ctx, _state()))
    assert "inconclusive:1" in out1["summary"]
    assert mem.get_hypothesis(hyp)["status"] == "supported"    # never overturned
    assert mem.get_hypothesis(hyp)["replication_attempts"] == 1

    out2 = asyncio.run(nodes.node_verify(ctx, _state()))       # retry #2
    assert "inconclusive:1" in out2["summary"]
    assert mem.get_hypothesis(hyp)["replication_attempts"] == 2

    out3 = asyncio.run(nodes.node_verify(ctx, _state()))       # budget spent
    assert out3["verifications"] == []
    assert mem.get_hypothesis(hyp)["replication_attempts"] == 2
    assert mem.get_hypothesis(hyp)["status"] == "supported"


def test_author_cannot_self_replicate(mem, tmp_path):
    _seed_supported(mem, "a01", "the method reaches >=0.9 precision")
    ctx = _ctx(mem, tmp_path, agent_id="a01")                 # the author
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert out["verifications"] == []
    assert sum(1 for e in mem.latest_evidence()
               if e["method"] == "replay") == 0


def test_replication_composes_with_a_fresh_finding(mem, tmp_path):
    """A cycle WITH a finding does both: replicates the settled claim AND
    verifies the new one — the results list carries both blocks."""
    _seed_supported(mem, "a01", "the method reaches >=0.9 precision")

    class PlanLLM(FakeLLM):
        async def ask(self, query, *, mode="concise", system=None,
                      purpose="general"):
            if purpose != "verify":
                return SimpleNamespace(answer="", citations=[], ok=False)
            payload = {"hypotheses": [{
                "claim": "novel ensemble gains >=0.85 on the toy set",
                "metric": "precision", "op": ">=", "target": 0.85,
                "protocol": PASS_PROTOCOL,
            }]}
            return SimpleNamespace(answer=json.dumps(payload),
                                   citations=[], ok=True)

    ctx = _ctx(mem, tmp_path, agent_id="a02", llm=PlanLLM())
    out = asyncio.run(nodes.node_verify(ctx, _state(
        finding_text="Finding: ensembles reach ~0.95 precision...")))
    assert "P24 replicated 1 claim(s)" in out["summary"]
    assert len(out["verifications"]) == 2
    assert out["verifications"][0]["replication"] is True
    assert out["verifications"][1]["replication"] is False
    assert out["verifications"][1]["status"] == "supported"
