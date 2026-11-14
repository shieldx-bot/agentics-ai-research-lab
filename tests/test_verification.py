"""P16 research side — hypothesis contract + register/run/settle lifecycle.

These tests compose the predecessor's primitives (verify_metrics.run_protocol,
SharedMemory hypotheses/evidence tables) the same way the live node will —
no LLM involved: the verdict must come from the evidence, whatever wrote the
protocol."""

import textwrap
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.memory.shared_memory import SharedMemory
from agentics.verify import (  # noqa: E402
    EXPERIMENT_DIR,
    HYPOTHESIS_PROMPT_CONTRACT,
    PROTOCOL_CONTRACT,
    Hypothesis,
    hypothesis_from_json,
    parse_hypothesis,
    register_hypothesis,
    run_hypothesis,
    settle_hypothesis,
    stable_id,
)
from agentics.verify_metrics import parse_metrics  # noqa: E402


# ------------------------------------------------------------- fixtures
@pytest.fixture()
def ws(tmp_path: Path):
    """A minimal local-only workspace (bare 'origin' + seed, like the real lab)."""
    import subprocess

    def git(cwd, *args):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-b", "main", str(seed))
    git(seed, "config", "user.email", "t@t")
    git(seed, "config", "user.name", "t")
    (seed / "README.md").write_text("seed\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "init")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "-q", "origin", "main")

    settings = Settings(github_token="x", github_repo="o/r")
    settings.data_dir = tmp_path / "data"
    w = RepoWorkspace("a02", settings)
    w._remote = str(bare)
    w.sync_main()
    return w


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _hyp(code: str, op: str = ">=", target: float = 0.9,
         metric: str = "precision", **over) -> Hypothesis:
    base = dict(
        hypothesis_id=stable_id("single-pass min/max equals naive scan"),
        claim="single-pass min/max equals naive scan",
        prediction="the measured equivalence rate stays above the target",
        metric=metric, op=op, target=target,
        code=textwrap.dedent(code).strip(),
        source_finding=7,
        task=3,
    )
    base.update(over)
    from agentics.verify_metrics import Criterion

    criterion = Criterion(metric=base.pop("metric"), op=base.pop("op"),
                          target=base.pop("target"))
    return Hypothesis(criterion=criterion, **base)


# ------------------------------------------------------------- contract
def test_parse_hypothesis_accepts_a_valid_proposal():
    raw = {
        "claim": "WAL mode improves write throughput",
        "prediction": "measured tps >= 100 on the synthetic workload",
        "criterion": {"metric": "tps", "op": ">=", "target": 100},
        "code": "print('AGENTICS_RESULT: {\"tps\": 120}')",
        "source_finding": 4,
        "task": 9,
    }
    h, why = parse_hypothesis(raw)
    assert h is not None and why == ""
    assert h.criterion.metric == "tps" and h.criterion.target == 100.0
    assert h.script_path.startswith(f"{EXPERIMENT_DIR}/")
    assert h.script_path.endswith(".py")
    assert HYPOTHESIS_PROMPT_CONTRACT in str(HYPOTHESIS_PROMPT_CONTRACT)
    assert "AGENTICS_RESULT" in PROTOCOL_CONTRACT


def test_parse_hypothesis_rejects_malformed_proposals_with_a_reason():
    good_code = "print('AGENTICS_RESULT: {\"x\": 1}')"
    bads = [
        ({}, "claim"),
        ({"claim": "c", "prediction": "p", "criterion": {"metric": "m", "op": "~", "target": 1},
          "code": good_code}, "criterion"),
        ({"claim": "c", "prediction": "p", "criterion": {"metric": "m", "op": ">=", "target": "abc"},
          "code": good_code}, "criterion"),
        ({"claim": "c", "prediction": "p", "criterion": {"metric": "m", "op": ">=", "target": 1},
          "code": "   "}, "code"),
        ({"claim": "c", "prediction": "", "criterion": {"metric": "m", "op": ">=", "target": 1},
          "code": good_code}, "prediction"),
        ({"claim": "c", "prediction": "p", "criterion": "metric>=1",
          "code": good_code}, "criterion"),
    ]
    for raw, needle in bads:
        h, why = parse_hypothesis(raw)
        assert h is None, raw
        assert needle in why


def test_hypothesis_from_json_roundtrip():
    h = _hyp("print(1)")
    h2, why = hypothesis_from_json(h.to_json())
    assert h2 is not None and why == ""
    assert h2.claim == h.claim and h2.criterion.as_dict() == h.criterion.as_dict()
    broken, why = hypothesis_from_json("{not json")
    assert broken is None and "invalid" in why


# ------------------------------------------------------------- run
def test_run_hypothesis_confirmed_from_real_execution(ws):
    code = textwrap.dedent("""
        import random
        random.seed(0)
        data = [random.random() for _ in range(500)]
        lo = hi = data[0]
        for x in data:
            if x < lo: lo = x
            if x > hi: hi = x
        ok = (min(data), max(data)) == (lo, hi)
        print(f'AGENTICS_RESULT: {{"precision": {1.0 if ok else 0.0}}}')
    """)
    ev = run_hypothesis(ws, _hyp(code, target=0.9))
    assert ev.verdict == "confirmed", ev.output
    assert ev.exit_code == 0 and ev.method == "protocol"
    assert ev.meta["metrics"]["precision"] == 1.0
    assert len(ev.meta["runs"]) == 2          # reproducibility: 2 runs
    assert (ws.path / f"{EXPERIMENT_DIR}").exists()


def test_run_hypothesis_refuted_when_metric_misses_target(ws):
    code = "print('AGENTICS_RESULT: {\"precision\": 0.42}')"
    ev = run_hypothesis(ws, _hyp(code, target=0.9))
    assert ev.verdict == "refuted"
    assert ev.meta["metrics"]["precision"] == 0.42


def test_run_hypothesis_inconclusive_on_crash(ws):
    ev = run_hypothesis(ws, _hyp("raise RuntimeError('boom')"))
    assert ev.verdict == "inconclusive"
    assert "boom" in ev.output


def test_run_hypothesis_inconclusive_without_marker(ws):
    ev = run_hypothesis(ws, _hyp("print('no marker here')"))
    assert ev.verdict == "inconclusive"
    assert "AGENTICS_RESULT" in ev.output


def test_run_hypothesis_inconclusive_when_not_reproducible(ws):
    code = textwrap.dedent("""
        import random
        print(f'AGENTICS_RESULT: {{"precision": {random.random()}}}')
    """)
    ev = run_hypothesis(ws, _hyp(code, target=0.5))
    assert ev.verdict == "inconclusive"
    assert "reproducible" in ev.output or "disagree" in ev.output


def test_run_hypothesis_inconclusive_when_guard_refuses_write(ws, monkeypatch):
    def refuse(path, content):
        raise ValueError("P10 protected path refused")

    monkeypatch.setattr(ws, "write_file", refuse)
    ev = run_hypothesis(ws, _hyp("print(1)"))
    assert ev.verdict == "inconclusive" and ev.exit_code == 126
    assert "guard" in ev.output


def test_parse_metrics_contract_helper():
    assert parse_metrics("noise\nAGENTICS_RESULT: {\"a\": 1.5}\n")["a"] == 1.5
    assert parse_metrics("nothing") == {}


# ------------------------------------------------------------- lifecycle
def _settled(mem, ws, code, **kw):
    h = _hyp(code, **kw)
    hyp_id = register_hypothesis(mem, "a02", h)
    ev = run_hypothesis(ws, h)
    # citations=1: an external anchor (the settle anchor law downgrades an
    # anchorless supported to `unanchored`; these tests cover the LIFECYCLE
    # mechanics, the law itself has its own suite)
    ev_id = settle_hypothesis(mem, "a02", hyp_id, h, ev, citations=1)
    return h, hyp_id, ev, ev_id


def test_lifecycle_pre_registers_before_the_run(mem):
    h = _hyp("print('AGENTICS_RESULT: {\"precision\": 1.0}')")
    hyp_id = register_hypothesis(mem, "a02", h)
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "testing"          # pre-registered, not settled yet
    assert row["criterion"]["metric"] == "precision"
    assert row["protocol_path"] == h.script_path
    # the swarm saw the registration while the protocol was still to run
    kinds = [m["kind"] for m in mem.recent_messages(limit=10)]
    assert "evidence" in kinds


def test_lifecycle_supported_roundtrip(mem, ws):
    code = "print('AGENTICS_RESULT: {\"precision\": 0.97}')"
    h, hyp_id, ev, ev_id = _settled(mem, ws, code, target=0.9)
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "supported"
    assert row["evidence_id"] == ev_id and row["attempts"] >= 1
    assert "0.97" in row["verdict_note"]
    ev_row = mem.evidence_for_task(h.task)[0]
    assert ev_row["hypothesis_id"] == hyp_id
    assert ev_row["metrics"]["precision"] == 0.97
    assert len(ev_row["runs"]) == 2
    # H3: verdict broadcast to every agent + settled claim in verified_knowledge
    joined = " ".join(m["content"] for m in mem.recent_messages(limit=10))
    assert "CONFIRMED" in joined
    vk = mem.verified_knowledge()
    assert any(v["claim"] == h.claim and v["status"] == "supported" for v in vk)


def test_lifecycle_refuted_claim_becomes_common_knowledge(mem, ws):
    h, hyp_id, ev, ev_id = _settled(
        mem, ws, "print('AGENTICS_RESULT: {\"precision\": 0.1}')", target=0.9)
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "refuted"
    vk = mem.verified_knowledge()
    assert any(v["claim"] == h.claim and v["status"] == "refuted" for v in vk)


def test_lifecycle_inconclusive_never_becomes_knowledge(mem, ws):
    h, hyp_id, ev, ev_id = _settled(mem, ws, "raise SystemExit(1)")
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "inconclusive"
    assert all(v["claim"] != h.claim for v in mem.verified_knowledge())
