"""P16bis — the researcher's verify node: claims become hypotheses, hypotheses
become MEASURED verdicts (H2), and verdicts land in shared knowledge (H3).

Fake LLM, real SharedMemory, real protocol execution through the workspace
runner — no network, no git needed (the workspace is a plain directory).
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
    "# tiny simulation: 200 predictions, ~90% correct\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('simulated eval done')\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
REFUTED_PROTOCOL = PROTOCOL.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")


class FakeLLM:
    account = SimpleNamespace(name="offline")

    def __init__(self, protocol: str, malformed_too: bool = False,
                 answer: str | None = None) -> None:
        self.protocol = protocol
        self.malformed_too = malformed_too
        self.answer = answer

    async def ask(self, query, *, mode="concise", system=None, purpose="general"):
        if purpose != "verify":
            return SimpleNamespace(answer="", citations=[], ok=False)
        if self.answer is not None:
            return SimpleNamespace(answer=self.answer, citations=[], ok=True)
        payload: dict = {"hypotheses": [
            {"claim": "the method reaches >=0.9 precision on the toy set",
             "metric": "precision", "op": ">=", "target": 0.9,
             "protocol": self.protocol},
        ]}
        if self.malformed_too:
            payload["hypotheses"].append(
                {"claim": "malformed one", "metric": "x", "op": "???",
                 "target": 1, "protocol": "print('hi')"})
        return SimpleNamespace(answer=json.dumps(payload), citations=[], ok=True)


class FakeWS:
    """A plain directory with the two methods node_verify uses; run_command
    executes workspace scripts for real (same allowlist contract: workspace .py)."""

    def __init__(self, root: Path, fail_write: bool = False) -> None:
        self.path = root
        self.fail_write = fail_write

    def sync_main(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_file(self, rel: str, content: str) -> None:
        if self.fail_write:
            raise ValueError("P10 protected path: blocked")
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        script = self.path / cmd[1]
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=timeout, cwd=self.path, check=False)
        return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _ctx(mem: SharedMemory, tmp_path: Path, llm, fail_write: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, memory=mem, llm=llm,
        workspace=FakeWS(tmp_path / "ws", fail_write=fail_write),
        cycle=1, settings=SimpleNamespace(research_topic="NFV", settle_require_anchor=False),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
    )


def _state(**over) -> dict:
    base = {"done": False, "finding_id": 5, "issue_number": None,
            "finding_text": "Finding: RF reaches ~0.9 precision..."}
    base.update(over)
    return base


def test_supported_hypothesis_reaches_verified_knowledge(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL))
    out = asyncio.run(nodes.node_verify(ctx, _state()))

    assert "supported:1" in out["summary"] and out["done"]
    hyp = mem.list_hypotheses()[0]
    assert hyp["status"] == "supported" and hyp["evidence_id"]
    assert hyp["criterion"] == {"metric": "precision", "op": ">=", "target": 0.9}
    ev = mem.latest_evidence()[0]
    assert ev["method"] == "protocol" and ev["verdict"] == "confirmed"
    assert ev["metrics"]["precision"] == pytest.approx(0.9, abs=0.05)
    assert ev["hypothesis_id"] == hyp["id"]
    assert len(ev["runs"]) == 2                       # rerun like a researcher
    knowledge = mem.verified_knowledge()
    assert [k["status"] for k in knowledge] == ["supported"]
    events = [j["event"] for j in mem.recent_journal()]
    assert "hypothesis_verified" in events
    protocol_files = list((tmp_path / "ws" / "experiments").glob("verify_h*.py"))
    assert len(protocol_files) == 1                   # artifact kept for review


def test_refuted_hypothesis_is_shared_and_not_sugarcoated(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(REFUTED_PROTOCOL))
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert "refuted:1" in out["summary"]
    knowledge = mem.verified_knowledge()
    assert knowledge[0]["status"] == "refuted"
    assert "measured precision" in knowledge[0]["note"]


def test_malformed_spec_is_inconclusive_and_the_rest_still_runs(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL, malformed_too=True))
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    statuses = {r["claim"][:12]: r["status"] for r in out["verifications"]}
    assert statuses == {"the method r": "supported", "malformed on": "inconclusive"}


def test_write_failure_never_fakes_success(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL), fail_write=True)
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert out["verifications"][0]["status"] == "inconclusive"
    assert mem.verified_knowledge() == []             # never becomes supported


def test_runner_error_is_inconclusive(mem, tmp_path):
    class BrokenWS(FakeWS):
        def run_command(self, cmd, timeout=120):
            raise RuntimeError("process group exploded")

    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL))
    ctx.workspace = BrokenWS(tmp_path / "ws")
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert out["verifications"][0]["status"] == "inconclusive"
    assert "runner error" in out["verifications"][0]["note"]


def test_no_finding_or_no_claims_skips_cleanly(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL))
    out = asyncio.run(nodes.node_verify(ctx, _state(finding_text="")))
    assert out["done"] and mem.list_hypotheses() == []
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL, answer='{"hypotheses": []}'))
    out = asyncio.run(nodes.node_verify(ctx, _state()))
    assert "no verifiable claim" in out["summary"] and mem.list_hypotheses() == []


def test_memory_context_carries_verified_knowledge(mem, tmp_path):
    ctx = _ctx(mem, tmp_path, FakeLLM(PROTOCOL))
    asyncio.run(nodes.node_verify(ctx, _state()))
    snap = nodes._memory_context(ctx)
    assert "[SUPPORTED]" in snap["verified"]
    fresh = nodes._memory_context(_ctx(mem, tmp_path, FakeLLM(PROTOCOL)))
    assert "nothing verified yet" not in snap["verified"]
    assert "[SUPPORTED]" in fresh["verified"]         # every agent sees it (H3)
