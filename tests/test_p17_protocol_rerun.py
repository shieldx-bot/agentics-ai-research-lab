"""P17 — protocol reproduction by the reviewer (composes on the P16 re-run).

An exit-code re-run (cmds_from_evidence + run_checks_on_branch) cannot judge
a QUANTITATIVE claim: a protocol always exits 0 — the frozen criterion is
the verdict. These tests pin the reviewer's full reproduction path:

* run_protocols_on_branch re-runs the author's protocol on origin/<branch>
  through the complete two-run adjudication against the ledger's criterion
  (a script that still prints AGENTICS_RESULT but misses the criterion is
  REFUTED — invisible to any exit-code re-run);
* a refuted reproduction re-verdicts the author's original evidence row
  (verdict_by trail), downgrades the hypothesis ledger entry, journals
  reviewer_rerun_refuted and blocks the approval — evidence overrides
  model opinion;
* cmd-path refuted re-runs re-verdict the author's original confirmed rows
  too, so the shared feed never keeps presenting a failed check as green;
* cmds_from_evidence only offers method='run_command' rows: protocol rows
  belong to this path, reviewer_rerun rows are never candidates.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.graph import nodes
from agentics.local_ops import LocalRepo
from agentics.memory.shared_memory import SharedMemory
from agentics.verify_metrics import Criterion
from agentics.verify_ops import Evidence, cmds_from_evidence

PY = sys.executable
PROTOCOL = "research_notes/experiments/h1.py"
BRANCH = "agent/a04/task-1-x"
CRIT = {"metric": "hit_rate", "op": ">=", "target": 0.5}


class FakeLLM:
    def __init__(self, answer: str):
        self.answer, self.account = answer, None

    async def ask(self, *args, **kwargs):
        return SimpleNamespace(answer=self.answer, citations=[])


class FakeReviewerWS:
    """Canned run_protocols_on_branch (+ optional run_checks_on_branch)."""

    def __init__(self, repro_evidence=None, check_results=None):
        self.repro_calls = []
        self.check_calls = []
        self._repro = repro_evidence or []
        self._checks = check_results

    def run_protocols_on_branch(self, branch, protocols, timeout=300):
        self.repro_calls.append((branch, [(p, c.describe()) for p, c in protocols]))
        return list(self._repro)

    def run_checks_on_branch(self, branch, cmds, timeout=300):
        self.check_calls.append((branch, [list(c) for c in cmds]))
        return [(cmd, code, out, 0.1) for cmd, code, out, _s in self._checks]


def _ctx(mem, ws, answer='{"verdict": "approve", "comment": "looks fine"}'):
    mq = SimpleNamespace(enqueued=[])
    mq.enqueue = lambda mr_id, author: mq.enqueued.append((mr_id, author))
    return SimpleNamespace(
        agent_id="a07", role=Role.REVIEWER, memory=mem, llm=FakeLLM(answer),
        meta={}, cycle=3, settings=SimpleNamespace(research_topic="NFV"),
        workspace=ws, merge_queue=mq, local=SimpleNamespace(guard=None),
        report=lambda **k: None,
    )


def _mr_with_protocol_evidence(mem, measured=0.72):
    task = mem.create_task("Add scan cache")
    mr_id = mem.open_review(
        task_id=task, title=f"[a04] Add scan cache (#{task})", body="b",
        author="a04", branch=BRANCH, files=[PROTOCOL],
        verification="tier-1 ok",
    )
    hyp_id = mem.add_hypothesis(
        "a04", "the shared scan cache halves discovery calls", CRIT,
        task_id=task, protocol_path=PROTOCOL, status="supported")
    ev_id = mem.record_evidence(
        "a04", task, f"{PROTOCOL} (hit_rate >= 0.5; 2 run(s))",
        0, "confirmed", hypothesis="the shared scan cache halves discovery calls",
        method="protocol", output=f"[measured hit_rate={measured} vs >= 0.5]",
        artifact=PROTOCOL)
    mem.attach_protocol_outcome(ev_id, {"hit_rate": measured}, [{"exit_code": 0}],
                                hypothesis_id=hyp_id)
    return task, mr_id, hyp_id, ev_id


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _repro_evidence(verdict: str, measured: float) -> Evidence:
    return Evidence(
        cmd=f"{PROTOCOL} (hit_rate >= 0.5; 2 run(s))", exit_code=0,
        verdict=verdict,
        output=f"[measured hit_rate={measured} vs hit_rate >= 0.5]",
        method="protocol",
        meta={"note": f"measured hit_rate={measured} vs hit_rate >= 0.5",
              "metrics": {"hit_rate": measured}, "runs": [{"exit_code": 0}] * 2,
              "criterion": CRIT})


# ------------------------------------------------------------ reproduction confirms
def test_protocol_reproduction_confirmed_supports_the_approval(mem):
    task, mr_id, hyp_id, ev_id = _mr_with_protocol_evidence(mem)
    ws = FakeReviewerWS(repro_evidence=[_repro_evidence("confirmed", 0.72)])
    state = asyncio.run(nodes.node_review(_ctx(mem, ws), {}))
    assert state["review_verdict"] == "approve"
    # reproduced on the MR branch, with the ledger's frozen criterion
    assert ws.repro_calls == [(BRANCH, [(PROTOCOL, "hit_rate >= 0.5")])]
    reruns = [e for e in mem.evidence_for_task(task) if e["method"] == "reviewer_rerun"]
    assert len(reruns) == 1 and reruns[0]["verdict"] == "confirmed"
    assert reruns[0]["artifact"] == PROTOCOL
    assert "P17 protocol reproduction" in state["review_comment"]
    # nothing was re-verdicted — the author's row stands confirmed
    row = [e for e in mem.evidence_for_task(task) if e["id"] == ev_id][0]
    assert row["verdict"] == "confirmed"


# ------------------------------------------------------------ reproduction refutes
def test_protocol_refuted_by_reproduction_blocks_the_approval(mem):
    """The case an exit-code re-run CANNOT catch: the script still exits 0,
    but the measured metric no longer meets the frozen criterion."""
    task, mr_id, hyp_id, ev_id = _mr_with_protocol_evidence(mem)
    ws = FakeReviewerWS(repro_evidence=[_repro_evidence("refuted", 0.31)])
    ctx = _ctx(mem, ws)
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["review_verdict"] == "request_changes"      # LLM 'approve' overridden
    assert ctx.merge_queue.enqueued == []                    # never merged
    assert "evidence overrides model opinion" in state["review_comment"]
    # the author's self-reported row is re-verdicted with a trail
    row = [e for e in mem.evidence_for_task(task) if e["id"] == ev_id][0]
    assert row["verdict"] == "refuted" and row["verdict_by"] == "a07"
    assert "reviewer reproduction failed" in row["note"]
    # the hypothesis ledger follows the machine refutation
    assert mem.get_hypothesis(hyp_id)["status"] == "refuted"
    assert any(j["event"] == "reviewer_rerun_refuted"
               for j in mem.recent_journal(limit=10))
    # the failed reproduction itself is shared evidence
    reruns = [e for e in mem.evidence_for_task(task) if e["method"] == "reviewer_rerun"]
    assert reruns and reruns[0]["verdict"] == "refuted"


def test_cmd_path_refuted_rerun_reverdicts_the_original_row(mem):
    """Extends the exit-code re-run: the author's original confirmed row must
    not keep showing green after the reviewer's re-run failed."""
    task = mem.create_task("Add retry budget")
    mem.open_review(
        task_id=task, title=f"[a04] retry budget (#{task})", body="b",
        author="a04", branch=BRANCH, files=["experiments/check.py"],
        verification="tier-1 ok")
    ev_id = mem.record_evidence(
        "a04", task, "python experiments/check.py", 0, "confirmed",
        hypothesis="retry budget cuts failures", method="run_command")
    ws = FakeReviewerWS(check_results=[
        (["python", "experiments/check.py"], 1, "assertion failed", 0.1)])
    state = asyncio.run(nodes.node_review(_ctx(mem, ws), {}))
    assert state["review_verdict"] == "request_changes"
    row = [e for e in mem.evidence_for_task(task) if e["id"] == ev_id][0]
    assert row["verdict"] == "refuted" and row["verdict_by"] == "a07"
    assert any(j["event"] == "reviewer_rerun_refuted"
               for j in mem.recent_journal(limit=10))


# ------------------------------------------------------------ candidate filtering
def test_cmds_from_evidence_offers_only_runnable_command_rows(mem):
    task = mem.create_task("mix of rows")
    mem.record_evidence("a04", task, "python a.py", 0, "confirmed",
                        method="run_command")
    mem.record_evidence("a04", task, f"{PROTOCOL} (hit_rate >= 0.5; 2 run(s))",
                        0, "confirmed", method="protocol")
    mem.record_evidence("a07", task, "python b.py", 0, "confirmed",
                        method="reviewer_rerun")
    out = cmds_from_evidence(mem.evidence_for_task(task), exclude_agent="a07")
    assert [c for c, _argv, _h in out] == ["python a.py"]


def test_inconclusive_reproduction_never_blocks(mem):
    task, mr_id, hyp_id, ev_id = _mr_with_protocol_evidence(mem)
    ws = FakeReviewerWS(repro_evidence=[_repro_evidence("inconclusive", 0.0)])
    state = asyncio.run(nodes.node_review(_ctx(mem, ws), {}))
    assert state["review_verdict"] == "approve"             # sandbox hiccup: no block
    row = [e for e in mem.evidence_for_task(task) if e["id"] == ev_id][0]
    assert row["verdict"] == "confirmed"                    # author's row stands


# ------------------------------------------------------------ real workspace
@pytest.fixture()
def lab(tmp_path):
    """A real bare origin + a main without the protocol + an MR branch that
    carries it — exactly the production situation: the protocol only exists
    on the author's branch."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)],
                   check=True, capture_output=True)
    (src / "docs").mkdir()
    (src / "docs" / "a.md").write_text("seed\n", encoding="utf-8")

    def git(*args):
        subprocess.run(["git", "-C", str(src), *args], check=True,
                       capture_output=True)

    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    git("checkout", "-b", BRANCH)
    proto = src / PROTOCOL
    proto.parent.mkdir(parents=True)
    proto.write_text('print(\'AGENTICS_RESULT: {"hit_rate": 0.72}\')\n',
                     encoding="utf-8")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-m", "protocol")
    repo = LocalRepo(settings)
    repo.ensure(src=src)
    git("push", "-u", str(repo.path), BRANCH)          # branch to origin
    ws = RepoWorkspace("a07", settings)
    ws._remote = str(repo.path)
    ws.ensure_cloned()
    ws.sync_main()
    return ws


def test_run_protocols_on_branch_reproduces_from_the_mr_branch(lab):
    """The protocol does NOT exist on main — the reproduction must check out
    origin/<branch>, adjudicate against the frozen criterion and restore a
    clean main afterwards."""
    assert not (lab.path / PROTOCOL).exists()           # main has no protocol
    repros = lab.run_protocols_on_branch(
        BRANCH, [(PROTOCOL, Criterion.from_dict(CRIT))], timeout=120)
    assert len(repros) == 1
    assert repros[0].verdict == "confirmed"
    assert repros[0].meta["metrics"]["hit_rate"] == 0.72
    assert repros[0].meta["runs"] and len(repros[0].meta["runs"]) == 2
    # workspace restored: back on a clean main, throw-away branch gone
    assert not (lab.path / PROTOCOL).exists()
    branch = subprocess.run(["git", "-C", str(lab.path), "branch", "--list", "review/tmp"],
                            capture_output=True, text=True).stdout.strip()
    assert branch == ""


def test_run_protocols_on_branch_refutes_a_missed_criterion(lab):
    repros = lab.run_protocols_on_branch(
        BRANCH, [(PROTOCOL, Criterion.from_dict(
            {"metric": "hit_rate", "op": ">=", "target": 0.9}))], timeout=120)
    assert repros[0].verdict == "refuted"               # 0.72 < 0.9
    assert repros[0].exit_code == 0                     # the script itself is green
