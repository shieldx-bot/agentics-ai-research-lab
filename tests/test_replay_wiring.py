"""P23 wiring — the supervisor's scan cycle replays settled claims.

The replay engine (agentics/verify/replay.py) existed with its unit tests,
but nothing in the LIVE cycle invoked it: replication stayed an owner-only
manual step, so drift was only caught when the owner remembered to run the
sweep. These tests pin the node_scan wiring: on-schedule sweeps really
re-run the stored protocols, off-schedule cycles skip them, and a falsified
claim is corrected in the swarm's shared knowledge with a broadcast every
agent sees (H3).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory

GOOD_CODE = (
    "import json\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
BAD_CODE = GOOD_CODE.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")


class FakeWS:
    """Plain-directory workspace; run_command executes workspace .py scripts
    for real (same contract tests/test_replay.py relies on)."""

    def __init__(self, root: Path) -> None:
        self.path = root

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        proc = subprocess.run([sys.executable, str(self.path / cmd[1])],
                              capture_output=True, text=True, timeout=timeout,
                              cwd=self.path, check=False)
        return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _scan_ctx(mem: SharedMemory, tmp_path: Path, cycle: int,
              sweep_size: int = 2, every_n: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="a01",
        role=Role.SUPERVISOR,
        memory=mem,
        claims=SimpleNamespace(sweep_stale=lambda: []),
        cycle=cycle,
        meta={"personal": None},
        settings=SimpleNamespace(
            research_topic="NFV",
            evolution_every_n_cycles=10,
            replay_sweep_size=sweep_size,
            replay_every_n_cycles=every_n,
        ),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=None,
        workspace=FakeWS(tmp_path / "ws"),
    )


def _settle(mem: SharedMemory, claim: str, code: str, status: str) -> int:
    return mem.add_hypothesis(
        "a02", claim, {"metric": "precision", "op": ">=", "target": 0.9},
        status=status, protocol_code=code)


def test_supervisor_scan_replays_settled_claims_on_schedule(mem, tmp_path):
    hyp_id = _settle(mem, "precision reaches 0.9", GOOD_CODE, "supported")
    state = nodes.node_scan(_scan_ctx(mem, tmp_path, cycle=3), {})
    replays = [e for e in mem.latest_evidence() if e["method"] == "replay"]
    assert len(replays) == 1 and replays[0]["verdict"] == "confirmed"
    assert "P23 replayed 1 claim(s)" in state["summary"]
    events = [j["event"] for j in mem.recent_journal(limit=20)]
    assert "claim_reconfirmed" in events
    # the re-settlement keeps the claim supported (reconfirm)
    assert mem.get_hypothesis(hyp_id)["status"] == "supported"


def test_supervisor_scan_skips_replay_off_schedule(mem, tmp_path):
    _settle(mem, "precision reaches 0.9", GOOD_CODE, "supported")
    state = nodes.node_scan(_scan_ctx(mem, tmp_path, cycle=1), {})
    assert not [e for e in mem.latest_evidence() if e["method"] == "replay"]
    assert "P23 replayed" not in state["summary"]


def test_supervisor_scan_disabled_by_zero_sweep_size(mem, tmp_path):
    _settle(mem, "precision reaches 0.9", GOOD_CODE, "supported")
    nodes.node_scan(_scan_ctx(mem, tmp_path, cycle=3, sweep_size=0), {})
    assert not [e for e in mem.latest_evidence() if e["method"] == "replay"]


def test_replay_falsifies_drifted_claim_and_broadcasts(mem, tmp_path):
    """A settled claim whose protocol no longer holds is corrected: the
    hypothesis flips to refuted, verified_knowledge stops advertising it as
    common ground, and the broadcast feed carries the REFUTED evidence."""
    hyp_id = _settle(mem, "precision reaches 0.9", BAD_CODE, "supported")
    assert mem.verified_knowledge()[0]["status"] == "supported"
    nodes.node_scan(_scan_ctx(mem, tmp_path, cycle=3), {})
    assert mem.get_hypothesis(hyp_id)["status"] == "refuted"
    # corrected: knowledge still lists the claim, but now as REFUTED common
    # ground ("never re-assert this") instead of supported fact
    assert mem.verified_knowledge()[0]["status"] == "refuted"
    events = [j["event"] for j in mem.recent_journal(limit=20)]
    assert "claim_falsified" in events
    broadcasts = " ".join(m["content"] for m in mem.recent_messages(limit=20))
    assert "REFUTED" in broadcasts


def test_broken_protocol_replay_never_breaks_the_scan(mem, tmp_path):
    """A protocol that no longer runs becomes claim_unstable + inconclusive —
    the sweep must still finish and the scan still reports its backlog."""
    _settle(mem, "precision reaches 0.9", "1 / 0\n", "supported")
    state = nodes.node_scan(_scan_ctx(mem, tmp_path, cycle=3), {})
    hyp = mem.list_hypotheses()[0]
    assert hyp["status"] == "inconclusive"
    assert "Backlog scan:" in state["summary"]          # scan completed
    assert state["done"] is True
