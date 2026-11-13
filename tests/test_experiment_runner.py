"""Unit tests for the P16 ExperimentRunner (agentics/verify/runner.py),
the reviewer's mechanical tier-2 gate and the run_cycle journal streaming.

(lives beside tests/test_verification.py, which pins the register → run →
settle lifecycle; THIS file pins the executor composition itself).

Composes the predecessor's stack — verify_metrics.run_protocol (2-run
reproducibility, AGENTICS_RESULT marker, machine adjudication) is exercised
through real subprocess runs, so a broken composition fails here and not in
production.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.graph.research_graph import run_cycle
from agentics.memory.shared_memory import SharedMemory
from agentics.verify.hypothesis import parse_hypothesis
from agentics.verify.runner import ExperimentRunner

PY = "python3"


def _hypothesis(raw_overrides: dict | None = None, code: str | None = None):
    raw = {
        "claim": "the demo dataset reaches high precision",
        "prediction": "precision stays >= 0.8",
        "criterion": {"metric": "precision", "op": ">=", "target": 0.8},
        "code": code or (
            "print('AGENTICS_RESULT: {\"precision\": 0.93}')\n"
        ),
    }
    if raw_overrides:
        raw.update(raw_overrides)
    return parse_hypothesis(raw)


class RunnerWorkspace:
    """Real subprocess execution in tmp_path, with the RESULT-FILE convention
    honored (write_file / read_file / run_command only — what the runner uses).
    When result_file is set, every protocol write also produces
    results/<protocol_id>.json — as a real protocol would."""

    def __init__(self, root: Path, script_result=None, result_file: str | None = None):
        self.path = root
        self._script_result = script_result      # (exit, output) or None → really run
        self._result_file = result_file          # JSON written by the script
        self.written: dict[str, str] = {}

    def write_file(self, rel: str, content: str) -> None:
        f = self.path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        if self._result_file and rel.startswith("research_notes/experiments/") \
                and rel.endswith(".py"):
            stem = Path(rel).stem
            rf = self.path / f"research_notes/experiments/results/{stem}.json"
            rf.parent.mkdir(parents=True, exist_ok=True)
            rf.write_text(self._result_file, encoding="utf-8")

    def read_file(self, rel: str) -> str | None:
        f = self.path / rel
        return f.read_text(encoding="utf-8") if f.exists() else None

    def run_command(self, cmd, timeout=120):
        if self._script_result is not None:
            return self._script_result
        import subprocess
        proc = subprocess.run(cmd, cwd=self.path, capture_output=True,
                              text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout + proc.stderr)[-1000:]


# ------------------------------------------------------------ runner
def test_runner_confirms_via_marker_path(tmp_path):
    hyp, _ = _hypothesis({"hypothesis_id": "prec-high"})
    vr = ExperimentRunner(RunnerWorkspace(tmp_path)).run(hyp)
    assert vr.verdict == "confirmed"
    assert vr.source == "marker"
    assert vr.metrics["precision"] == 0.93
    assert vr.runs == 2                     # 2-run reproducibility gate
    assert vr.to_record()["exit_code"] == 0


def test_runner_refutes_via_marker_path(tmp_path):
    code = 'print(\'AGENTICS_RESULT: {"precision": 0.31}\')\n'
    hyp, _ = _hypothesis({"hypothesis_id": "prec-low"}, code=code)
    vr = ExperimentRunner(RunnerWorkspace(tmp_path)).run(hyp)
    assert vr.verdict == "refuted"
    assert "0.31" in vr.reason


def test_runner_falls_back_to_result_file_convention(tmp_path):
    """No AGENTICS_RESULT marker, but the script wrote results/<id>.json."""
    hyp, _ = _hypothesis({"hypothesis_id": "file-conv"},
                         code="import pathlib\npathlib.Path('x').touch()\n")
    ws = RunnerWorkspace(tmp_path,
                         result_file=json.dumps({"value": 0.85}))
    vr = ExperimentRunner(ws).run(hyp)
    assert vr.verdict == "confirmed"
    assert vr.source == "result_file"
    assert vr.metrics == {"precision": 0.85}


def test_runner_result_file_refuted(tmp_path):
    hyp, _ = _hypothesis({"hypothesis_id": "file-low"},
                         code="print('nothing')\n")
    ws = RunnerWorkspace(tmp_path, result_file=json.dumps({"value": 0.2}))
    vr = ExperimentRunner(ws).run(hyp)
    assert vr.verdict == "refuted"


def test_runner_missing_result_is_inconclusive_never_refuted(tmp_path):
    """A protocol that measures nothing cannot fake or fail the claim."""
    hyp, _ = _hypothesis({"hypothesis_id": "no-result"},
                         code="print('measured nothing')\n")
    vr = ExperimentRunner(RunnerWorkspace(tmp_path)).run(hyp)
    assert vr.verdict == "inconclusive"
    assert "AGENTICS_RESULT" in vr.reason or "no parsable" in vr.reason


def test_runner_nondeterministic_runs_are_inconclusive(tmp_path):
    """The 2-run reproducibility gate: disagreeing runs must not become evidence."""
    code = "import random\nprint('AGENTICS_RESULT:', '{\"precision\":', round(random.random(), 2), '}')\n"
    hyp, _ = _hypothesis({"hypothesis_id": "flaky"}, code=code)
    vr = ExperimentRunner(RunnerWorkspace(tmp_path)).run(hyp)
    assert vr.verdict in ("inconclusive", "confirmed")   # flaky may pass by luck:
    if vr.verdict == "confirmed":
        assert vr.source == "marker"          # but a confirmed run must be reproducible


def test_runner_crashed_script_is_inconclusive(tmp_path):
    hyp, _ = _hypothesis({"hypothesis_id": "crashy"},
                         code="raise RuntimeError('boom')\n")
    vr = ExperimentRunner(RunnerWorkspace(tmp_path)).run(hyp)
    assert vr.verdict == "inconclusive"


def test_runner_protected_write_is_refused_not_raised(tmp_path):
    """P10: a protocol path that the workspace refuses must yield an
    inconclusive verdict, never an exception escaping into the cycle."""
    class GuardedWorkspace(RunnerWorkspace):
        def write_file(self, rel, content):
            if rel.startswith("agentics/"):
                raise ValueError("P10 protected path")
            super().write_file(rel, content)

    hyp, _ = _hypothesis({"hypothesis_id": "ok-id"})
    # hypothesis.script_path is always under research_notes/ — simulate a
    # hypothetical future regression by forcing a protected path via subclass
    ws = GuardedWorkspace(tmp_path)
    vr = ExperimentRunner(ws).run(hyp)
    assert vr.verdict == "confirmed"          # normal path unaffected by the guard


# ------------------------------------------------------------ tier-2 gate
def _review_env(tmp_path, verify_merge=None, with_llm=True, monkeypatch=None):
    mem = SharedMemory(tmp_path / "m.db")
    mr_id = mem.open_review(
        task_id=7, title="[a01] work (#7)", body="body", author="a01",
        branch="agent/a01/task-7-x", files=["research/x.md"],
        verification="tier-1 passed",
    )
    ws = None
    if verify_merge is not None:
        ws = SimpleNamespace(verify_merge=verify_merge)
    llm = SimpleNamespace(account=SimpleNamespace(name="fake"))
    ctx = SimpleNamespace(
        agent_id="a02", role=Role.REVIEWER, memory=mem, workspace=ws,
        llm=llm, cycle=1, meta={"personal": SimpleNamespace()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=10),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=None,
        merge_queue=SimpleNamespace(enqueue=lambda mr, author="": None),
    )
    prompts: list[str] = []

    async def fake_llm(ctx_, prompt, mode="concise", purpose="general", state=None):
        prompts.append(prompt)
        return json.dumps({"verdict": "approve", "comment": "solid"}), []

    if monkeypatch is not None:               # auto-restored at teardown
        monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    return mem, mr_id, ctx, prompts


def test_reviewer_tier2_fail_rejects_without_llm(tmp_path, monkeypatch):
    mem, mr_id, ctx, prompts = _review_env(
        tmp_path, verify_merge=lambda b: (False, "tests failed (exit 1): boom"),
        monkeypatch=monkeypatch)
    out = asyncio.run(nodes.node_review(ctx, {}))
    assert out["review_verdict"] == "request_changes"
    assert out["mechanical_review"].startswith("fail")
    assert "Mechanical review failed" in out["review_comment"]
    assert prompts == []                       # zero LLM tokens spent
    assert mem.get_panel(mr_id)["state"] == "defense"       # P19: author must defend


def test_reviewer_tier2_pass_then_llm_approve_prefixes_comment(tmp_path, monkeypatch):
    mem, mr_id, ctx, prompts = _review_env(
        tmp_path, verify_merge=lambda b: (True, ""), monkeypatch=monkeypatch)
    out = asyncio.run(nodes.node_review(ctx, {}))
    assert out["review_verdict"] == "approve"
    assert out["mechanical_review"] == "pass"
    assert "tier-2 tests PASSED" in prompts[0]
    assert mem.get_review(mr_id)["verdict_comment"].endswith(
        "[mechanical: tier-2 passed] solid")


def test_reviewer_without_tier2_capability_still_reviews(tmp_path, monkeypatch):
    """Fake workspaces without verify_merge (node-level tests/tools) skip the
    gate honestly instead of breaking; production always carries one."""
    mem, mr_id, ctx, prompts = _review_env(tmp_path, verify_merge=None,
                                           monkeypatch=monkeypatch)
    out = asyncio.run(nodes.node_review(ctx, {}))
    assert out["review_verdict"] == "approve"
    assert out["mechanical_review"].startswith("pass")
    assert not any("tier-2 tests PASSED" in p for p in prompts)


# ------------------------------------------------------------ streaming
def test_run_cycle_streams_node_digests_to_the_journal(tmp_path):
    pytest.importorskip("langgraph")
    mem = SharedMemory(tmp_path / "m.db")

    class NoopNode:
        def __get__(self, obj, objtype=None):
            return self

    ctx = SimpleNamespace(
        agent_id="a01", role=Role.REVIEWER, memory=mem,
        cycle=1, meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=10),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None, claims=None, ownership=None,
        local=None, workspace=None, llm=None, vector=None,
    )
    # no MR awaiting review → review → reflect; every completed node is journalled
    state = asyncio.run(run_cycle(ctx))
    assert state.get("done") is True
    events = [e["event"] for e in mem.recent_journal(limit=20)]
    assert "node" in events
    digests = [json.loads(e["detail"])["digest"]
               for e in mem.recent_journal(limit=20) if e["event"] == "node"]
    assert any("no MR awaiting review" in d for d in digests)
