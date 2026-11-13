"""P16 bridge — the hypothesis contract (agentics/verify, by another worker)
runs INSIDE the research cycle and its verdicts land in the SHARED evidence
table (agentics/memory, P16 ledger). Together they answer the owner's three
hypotheses end-to-end: an agent proposes a measurable claim, RUNS real code,
the runner re-judges the criterion from the raw measurement, and the verdict
becomes common knowledge (reviewer prompt, dashboard, CLI).

The runner was unit-tested in tests/test_verification.py but was not wired
into the graph — these tests pin the wiring and the gates:
  * malformed hypothesis  -> journaled, never guessed through, cycle lives on
  * confirmed measurement -> evidence row (method=hypothesis_contract), pushed
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

_VALID_RAW = {
    "hypothesis_id": "cache-hit-rate",
    "claim": "the shared scan cache halves discovery calls",
    "prediction": "hit rate stays above 50% across scans",
    "metric": "hit_rate",
    "comparison": ">=",
    "threshold": 0.5,
    # run_protocol contract: the protocol PRINTS its measurement as one final
    # AGENTICS_RESULT line (deterministic — the runner runs it twice and
    # non-reproducible output is inconclusive, never silently accepted).
    "code": (
        "import json\n"
        "print('AGENTICS_RESULT: ' + json.dumps({'hit_rate': 0.72}))\n"
    ),
}


class BridgeWorkspace:
    """Fake with real file semantics: write_file/read_file touch tmp_path,
    run_command is canned per stage (py_compile/pytest/script)."""

    def __init__(self, tmp_path: Path, changed=("experiments/a.py",),
                 script_result=(0, "ok")):
        self.path = tmp_path
        self._changed = list(changed)
        self.script_result = script_result
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
        # experiments really execute (cwd = workspace) so the contract's
        # result JSON lands on disk and the runner can judge from it
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
    assert row["method"] == "hypothesis_contract"
    assert "cache-hit-rate.py" in row["cmd"] and "hit_rate" in row["cmd"]
    # artifact = the protocol script itself (re-runnable by any other agent)
    assert row["artifact"].endswith("cache-hit-rate.py")
    assert "hit_rate=0.72" in row["output"]
    assert row["metrics"] == {"hit_rate": 0.72}      # attach_protocol_outcome
    assert len(row["runs"]) == 2                     # reproducibility: 2 runs
    assert any(m["kind"] == "evidence" for m in mem.recent_messages())


def test_settlement_supported_claim_becomes_common_ground(tmp_path):
    """H3 closed loop: pre-register -> run -> settle with evidence_id -> the
    claim shows up in verified_knowledge() that every agent's prompt carries."""
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
    raw = dict(_VALID_RAW, code=_VALID_RAW["code"].replace("0.72", "0.31"))
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert not ws.pushed
    assert "REFUTED" in state.get("yield_reason", "")
    assert mem.evidence_for_task(1)[0]["verdict"] == "refuted"
    assert mem.get_hypothesis(
        mem.list_hypotheses()[0]["id"])["status"] == "refuted"


def test_malformed_hypothesis_is_journaled_never_guessed(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    raw = dict(_VALID_RAW, comparison="≈")            # not in the allowed set
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert ws.pushed                                   # cycle survives
    assert mem.latest_evidence() == []                 # nothing recorded as evidence
    assert mem.list_hypotheses() == []                 # never pre-registered either
    assert any(j["event"] == "p16_hypothesis_invalid"
               for j in mem.recent_journal(limit=10))


def test_missing_result_json_is_inconclusive_not_refuted(tmp_path):
    """A protocol that prints no AGENTICS_RESULT line cannot fake or fail the
    claim — a missing measurement is recorded as inconclusive, never as
    success."""
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    raw = dict(_VALID_RAW, code="print('measured nothing')\n")
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[raw])))
    assert ws.pushed                                   # no livelock
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
    assert methods == {"run_command", "hypothesis_contract"}
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
    assert "research_notes/experiments/cache-hit-rate.py" in state["changed_files"]
