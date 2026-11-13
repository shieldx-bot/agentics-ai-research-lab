"""P16 bridge — the hypothesis contract runs INSIDE the research cycle.

Composes the verify package's register -> run -> settle glue (which itself
composes verify_metrics' two-run reproducibility + machine adjudication and
the shared evidence/hypotheses tables). These tests pin the coder-path
wiring on node_implement:

  * malformed hypothesis  -> journaled, never guessed through, cycle lives on
  * confirmed measurement -> evidence row (method=protocol), pushed,
                            hypothesis settled 'supported' with evidence_id
  * refuted measurement   -> yield before the push (a falsified claim must
                             not reach main)
  * inconclusive          -> journaled, reviewer decides, no livelock
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard

PY = sys.executable

_RESULT = 'print(\'AGENTICS_RESULT: {"hit_rate": 0.72}\')\n'

_VALID_RAW = {
    "hypothesis_id": "cache-hit-rate-h1",
    "claim": "the shared scan cache halves discovery calls",
    "prediction": "hit rate stays above 50% across scans",
    "criterion": {"metric": "hit_rate", "op": ">=", "target": 0.5},
    "code": (
        "import time\n"
        "time.sleep(0.01)\n"        # deterministic, measurable, fast
        + _RESULT
    ),
}


class BridgeWorkspace:
    """Fake with real file + subprocess semantics: scripts actually run
    (cwd = workspace) so AGENTICS_RESULT reaches stdout and the protocol
    judge runs on real output; -m commands are canned (tier-1 passes)."""

    def __init__(self, tmp_path: Path, changed=("experiments/a.py",)):
        self.path = tmp_path
        self._changed = list(changed)
        self.pushed, self.cmds = [], []
        for f in self._changed:
            (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / f).write_text("x = 1\n")

    def sync_main(self): ...
    def checkout_new_branch(self, b): ...
    def changed_files(self): return list(self._changed)
    def read_file(self, p):
        f = self.path / p
        return f.read_text() if f.exists() else None

    def write_file(self, p, c):
        f = self.path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(c, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        self.cmds.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "-m":
            return (0, "ok")                     # tier-1 compile/pytest pass
        import subprocess
        try:
            proc = subprocess.run(cmd, cwd=self.path, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        return proc.returncode, (proc.stdout + proc.stderr)[-1000:]

    def commit_and_push(self, branch, msg, allow_protected=False):
        self.pushed.append(branch)
        return True


def _impl_ctx(mem, ws):
    return SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem,
        claims=SimpleNamespace(release=lambda *a, **k: None),
        cycle=1, meta={"personal": SimpleNamespace()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=10),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=SimpleNamespace(guard=ProtectedPathsGuard()),
        ownership=SimpleNamespace(
            evaluate=lambda role, changed, active: (True, [])),
        workspace=ws,
    )


def _impl_state(**kw):
    state = {
        "branch": "agent/a04/task-1-x", "issue_number": 1, "issue_title": "t",
        "category": "code", "plan": json.dumps({"files": [], "commit_message": "m"}),
    }
    state.update(kw)
    return state


def test_confirmed_hypothesis_reaches_the_shared_ledger(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[_VALID_RAW])))
    assert ws.pushed
    rows = mem.evidence_for_task(1)
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "confirmed"
    assert row["method"] == "protocol"
    assert "cache-hit-rate-h1.py" in row["cmd"]
    assert row["artifact"].endswith("cache-hit-rate-h1.py")
    assert row["exit_code"] == 0
    assert any(m["kind"] == "evidence" for m in mem.recent_messages())


def test_settlement_supported_claim_becomes_common_ground(tmp_path):
    """H3 closed loop: pre-register -> run twice -> settle with evidence_id ->
    the claim shows up in verified_knowledge() every agent's prompt carries."""
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[_VALID_RAW])))
    knowledge = mem.verified_knowledge()
    assert len(knowledge) == 1
    k = knowledge[0]
    assert k["status"] == "supported" and "scan cache" in k["claim"]
    assert k["evidence_id"] == mem.evidence_for_task(1)[0]["id"]
    assert k["criterion"]["metric"] == "hit_rate" and k["criterion"]["target"] == 0.5


def test_refuted_hypothesis_blocks_the_push(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    raw = dict(_VALID_RAW, code=_RESULT.replace("0.72", "0.31"))
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert not ws.pushed
    assert "REFUTED" in state.get("yield_reason", "")
    assert mem.evidence_for_task(1)[0]["verdict"] == "refuted"
    assert mem.verified_knowledge()[0]["status"] == "refuted"


def test_malformed_hypothesis_is_journaled_never_guessed(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    raw = dict(_VALID_RAW, criterion={"metric": "hit_rate", "op": "≈", "target": 0.5})
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert ws.pushed                                   # cycle survives
    assert mem.latest_evidence() == []                 # nothing recorded as evidence
    assert any(j["event"] == "p16_hypothesis_invalid"
               for j in mem.recent_journal(limit=10))


def test_missing_result_marker_is_inconclusive_not_refuted(tmp_path):
    """A protocol that prints no AGENTICS_RESULT line cannot fake or fail the
    claim: inconclusive, journal note, reviewer decides, no livelock."""
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    raw = dict(_VALID_RAW, code="print('measured nothing')\n")
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert ws.pushed
    row = mem.evidence_for_task(1)[0]
    assert row["verdict"] == "inconclusive"
    assert any(j["event"] == "p16_inconclusive"
               for j in mem.recent_journal(limit=10))


def test_verify_cmds_and_hypotheses_coexist(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    ws.write_file("exp.py", "print('ok')\n")           # the declared check exists
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(
            verify_cmds=[[PY, "exp.py"]], hypotheses=[_VALID_RAW])))
    assert ws.pushed
    methods = {r["method"] for r in mem.evidence_for_task(1)}
    assert methods == {"run_command", "protocol"}
    assert "P16" in state["verification"]


def test_protocol_file_joins_the_mr_record(tmp_path):
    """The pushed branch carries the protocol file (git add -A); the MR record
    must list it too, or the reviewer sees a file in the diff that the record
    does not explain — the protocol IS the evidence's source code."""
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[_VALID_RAW])))
    assert ws.pushed
    assert "research_notes/experiments/cache-hit-rate-h1.py" in state["changed_files"]
