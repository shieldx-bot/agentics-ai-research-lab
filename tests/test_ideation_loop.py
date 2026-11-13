"""QD ideation closed loop (2026-11-14 "integrate deeper" round).

Mechanisms under test, one per source project:
- IDEAgent (arXiv 2607.22375): judge + qualification gate (tau=60, repair
  margin 20), rejected-pattern archive (R store), repair-within-lineage.
- Kosmos (arXiv 2511.02824): insight harvest — a settled experiment
  proposes its follow-ups; gap proposal — the world model proposes the
  next experiment when topic + registry are both exhausted.
- autoresearch (karpathy/autoresearch): experiment ledger (every settle,
  keep/discard semantics), bounded near-miss retry with the failure note,
  retire-after-failed-retry.
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


# ------------------------------------------------------------ pure gate math

def test_quality_score_weighting_prefers_non_obviousness():
    j = idea.Judgement(non_obviousness=80, clarity=70, feasibility=50,
                       soundness=7)
    # 0.7*80 + 0.2*(7/9*100) + 0.1*70 = 56 + 15.56 + 7
    assert abs(j.quality() - 78.6) < 0.2
    # soundness 9/9 contributes only 20 points — non-obviousness dominates
    perfect = idea.Judgement(100, 100, 100, 9)
    assert perfect.quality() == pytest.approx(100.0)


def test_qualification_routes_accept_repair_reject():
    accept = idea.Judgement(70, 65, 40, 6)
    assert idea.qualification(accept) == "accept"
    # every gated score within 20 pts below tau -> ONE repair
    repair = idea.Judgement(50, 65, 40, 6)          # N=50 (gap 10)
    assert idea.qualification(repair) == "repair"
    # weakest axis below the margin -> straight reject
    reject = idea.Judgement(35, 65, 40, 6)          # N=35 (gap 25)
    assert idea.qualification(reject) == "reject"
    # soundness 0-9 rescales to 100: score 5 -> 55.6 (repair band)
    borderline = idea.Judgement(70, 70, 40, 5)
    assert idea.qualification(borderline) == "repair"


def test_parse_children_validates_contract():
    good = json.dumps({"hypotheses": [
        {"claim": "a GNN forecaster cuts VNF migration count under bursts",
         "metric": "migration_count", "op": "<", "target": 3,
         "tolerance": 0.1,
         "signature": {"problem": "p", "mechanism": "m", "novelty": "n",
                       "assumptions": "a", "effect": "e"}},
        {"claim": "too short", "metric": "m", "op": ">=", "target": 1},
        {"claim": "bad operator entry here for testing purposes",
         "metric": "m", "op": "==", "target": 1},
        {"claim": "missing metric entirely in this entry", "op": ">=",
         "target": 1},
        {"claim": "non numeric target in this hypothesis entry",
         "metric": "m", "op": ">=", "target": "high"},
        "not a dict",
    ]})
    kids = idea.parse_children(good, max_children=2)
    assert len(kids) == 1
    assert kids[0]["criterion"]["op"] == "<"
    assert kids[0]["criterion"]["tolerance"] == 0.1
    assert kids[0]["signature"]["mechanism"] == "m"
    assert idea.parse_children("no json at all") == []
    assert idea.parse_children('{"other": 1}') == []


def test_parse_scores_clamps_and_rejects():
    raw = json.dumps({"non_obviousness": 850, "clarity": -3, "feasibility": 66,
                      "soundness": 8, "notes": "sharp but wordy"})
    j = idea.parse_scores(raw)
    assert j is not None and j.non_obviousness == 100 and j.clarity == 0.0
    assert j.soundness_100() == pytest.approx(8 / 9 * 100)
    assert idea.parse_scores("prose only") is None
    assert idea.parse_scores('{"non_obviousness": "high"}') is None


def test_parse_repair_returns_single_child():
    raw = json.dumps({"claim": "a sharpened reformulation of the mechanism",
                      "metric": "rmse", "op": "<=", "target": 5.0})
    child = idea.parse_repair(raw)
    assert child is not None and child["criterion"]["target"] == 5.0
    two = json.dumps({"hypotheses": [
        {"claim": "first child claim of two", "metric": "m", "op": ">=",
         "target": 1},
        {"claim": "second child claim here", "metric": "m", "op": ">=",
         "target": 1}]})
    assert idea.parse_repair(two)["claim"] == "first child claim of two"


# ------------------------------------------------------------------- memory

@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _hyp(mem, claim, status="proposed", code="", attempts=None):
    hid = mem.add_hypothesis("a02", claim,
                             {"metric": "m", "op": ">=", "target": 1},
                             status=status, protocol_code=code)
    if attempts is not None:
        with mem._lock, mem._conn:
            mem._conn.execute(
                "UPDATE hypotheses SET attempts=? WHERE id=?", (attempts, hid))
    return hid


def test_experiment_ledger_roundtrip_and_prune(mem):
    mem.record_experiment("a01", 1, "supported", metric="precision",
                          measured=0.93, op=">=", target=0.9,
                          verdict_note="measured 0.93", protocol_path="x.py",
                          source="deep_dive", description="keep me")
    mem.record_experiment("a02", 2, "refuted", metric="precision",
                          measured=0.10, op=">=", target=0.9, source="coder")
    rows = mem.ledger_recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["status"] == "refuted"          # newest first
    assert rows[1]["status"] == "supported"
    assert rows[1]["measured"] == pytest.approx(0.93)
    assert rows[1]["hyp_id"] == 1
    for i in range(600):
        mem.record_experiment("a01", None, "inconclusive")
    with mem._lock:
        n = mem._conn.execute(
            "SELECT COUNT(*) FROM experiment_ledger").fetchone()[0]
    assert n == 500                                # bounded world model rule


def test_rejected_patterns_render_and_prune(mem):
    mem.add_rejected_pattern("a01", "threshold autoscaler clone",
                             "autoscaling", "near-duplicate")
    mem.add_rejected_pattern("a01", "LSTM forecasting variant",
                             "forecasting", "gate rejected")
    text = mem.rejected_render()
    assert "autoscaling" in text and "forecasting" in text
    assert "(rejected: near-duplicate)" in text
    for i in range(110):
        mem.add_rejected_pattern("a01", f"pattern {i}", "other", "x")
    with mem._lock:
        n = mem._conn.execute(
            "SELECT COUNT(*) FROM rejected_patterns").fetchone()[0]
    assert n == 100                                # compact by contract


def test_testable_hypotheses_retries_first_excludes_exhausted(mem):
    _hyp(mem, "fresh proposed claim about autoscaling policies")
    _hyp(mem, "inconclusive retryable claim about LSTM forecasts",
         status="inconclusive", attempts=1)
    _hyp(mem, "inconclusive exhausted claim about scheduling queues",
         status="inconclusive", attempts=2)
    got = mem.testable_hypotheses(limit=6, max_attempts=2)
    ids = [h["id"] for h in got]
    assert len(ids) == 2                           # exhausted one excluded
    assert got[0]["status"] == "inconclusive"      # retry first
    assert got[0]["attempts"] == 1
    assert got[1]["status"] == "proposed"


# ------------------------------------------------------------- harness bits

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
    # legacy mode: the anchor law + double-judge gate have their own suite
    settle_require_anchor = False
    discovery_second_judge = False


GOOD_JUDGE = json.dumps({"non_obviousness": 85, "clarity": 80,
                         "feasibility": 70, "soundness": 8,
                         "notes": "solid mechanism, clean test path"})
LOW_JUDGE = json.dumps({"non_obviousness": 30, "clarity": 40,
                        "feasibility": 60, "soundness": 4,
                        "notes": "routine ablation of known work"})
NEAR_MISS_JUDGE = json.dumps({"non_obviousness": 50, "clarity": 65,
                              "feasibility": 60, "soundness": 6,
                              "notes": "novel but vaguely specified"})


class FakeWS:
    """Workspace double: real subprocess runs, same allowlist contract."""

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
    """Purpose-routed fake: answers are (marker, payload) pairs matched
    against the prompt body; the FIRST match is consumed (pop) so repeated
    markers (judge #1 vs judge #2) can return different payloads."""

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


def _child(claim="a GNN topology forecaster cuts VNF migration count "
                 "under bursty three-VNF traffic"):
    # the IDEATOR wire contract: metric/op/target (+ optional quintuple) —
    # the same shape parse_children validates.
    return {"claim": claim, "metric": "migration_count", "op": "<",
            "target": 3.0,
            "signature": {"problem": "migration storms",
                          "mechanism": "GNN topology forecasting",
                          "novelty": "topology-aware trigger",
                          "assumptions": "graph stability",
                          "effect": "fewer migrations"}}


# ------------------------------------------------- IDEAgent gate pipeline

def test_gate_and_register_accepts_and_links_parent(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    ctx = _ctx(mem, tmp_path, FakeLLM(("ROLE: judge", GOOD_JUDGE)))
    parent = _hyp(mem, "settled parent claim about autoscaling policies")
    recent = mem.recent_hypotheses(12)
    hyp_id = asyncio.run(
        _gate_and_register(ctx, _child(), [parent], recent))
    assert hyp_id is not None
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "proposed"
    assert row["parents"] == [parent]              # Kosmos lineage
    sig = mem.recent_hypotheses(3)[0]["signature"]
    assert sig["mechanism"] == "GNN topology forecasting"
    events = [e["event"] for e in mem.recent_journal(limit=5, agent="a02")]
    assert "idea_registered" in events


def test_gate_and_register_pem_rejects_clone_without_llm(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    prior = ("a linear regression model estimates VNF CPU utilization from "
             "packet rate with held-out RMSE under 5 points")
    _hyp(mem, prior)
    llm = FakeLLM()
    ctx = _ctx(mem, tmp_path, llm)
    clone = _child(claim="a linear regression performance model estimates "
                         "VNF CPU utilization from packet rate, held-out "
                         "RMSE under 5 points")
    recent = mem.recent_hypotheses(12)
    hyp_id = asyncio.run(_gate_and_register(ctx, clone, [], recent))
    assert hyp_id is None
    assert llm.calls == []                         # mechanical gate, no ask
    text = mem.rejected_render()
    assert "near-duplicate" in text and "regression" in text
    events = [e["event"] for e in mem.recent_journal(limit=5, agent="a02")]
    assert "idea_rejected" in events


def test_gate_and_register_repair_path_promotes_fixed_child(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    fixed = json.dumps({
        "claim": "a topology-aware GNN forecaster cuts VNF migration count "
                 "in bursty three-VNF chains by predicting link utilization",
        "metric": "migration_count", "op": "<", "target": 3.0})
    # judge is consumed twice: first -> near-miss (repair), second -> accept
    llm = FakeLLM(("ROLE: repair", fixed),
                  ("ROLE: judge", NEAR_MISS_JUDGE),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    child = _child(claim="a GNN forecaster reduces VNF migration counts "
                         "under bursty traffic patterns")
    hyp_id = asyncio.run(_gate_and_register(ctx, child, [], []))
    assert hyp_id is not None
    row = mem.get_hypothesis(hyp_id)
    assert "topology-aware" in row["claim"]        # repaired child stored
    events = [e["event"] for e in mem.recent_journal(limit=8, agent="a02")]
    assert "idea_repaired" in events


def test_gate_and_register_rejects_low_scores(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    ctx = _ctx(mem, tmp_path, FakeLLM(("ROLE: judge", LOW_JUDGE)))
    child = _child()
    hyp_id = asyncio.run(_gate_and_register(ctx, child, [], []))
    assert hyp_id is None
    assert mem.rejected_render() != ""             # archived as pattern
    assert "routine" in mem.rejected_render() or "gate rejected" \
        in mem.rejected_render()


# ------------------------------------------------------- Kosmos harvest

def test_harvest_insights_registers_children_of_settled(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim about autoscaling policies")
    ideator = json.dumps({"hypotheses": [
        _child(),                                  # fresh -> registered
        {"claim": "a GNN topology forecaster cuts VNF migration count "
                  "under bursty three-VNF traffic",   # clone of child #1
         "metric": "migration_count", "op": "<", "target": 3},
    ]})
    llm = FakeLLM(("ROLE: ideator", ideator),
                  ("ROLE: judge", GOOD_JUDGE),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    n = asyncio.run(_harvest_insights(ctx, parent, "parent claim",
                                      "supported", "precision=0.93"))
    assert n == 1
    rows = mem.list_hypotheses(limit=5)
    kids = [r for r in rows if r["parents"] == [parent]]
    assert len(kids) == 1
    events = [e["event"] for e in mem.recent_journal(limit=8, agent="a02")]
    assert "insight_harvest" in events
    assert "idea_rejected" in events               # the clone was archived


def test_harvest_disabled_by_settings(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    llm = FakeLLM(("ROLE: ideator", '{"hypotheses": []}'))
    ctx = _ctx(mem, tmp_path, llm)
    ctx.settings = SimpleNamespace(**{**vars(_S()), "ideation_enabled": False})
    n = asyncio.run(_harvest_insights(ctx, 1, "claim", "supported", "x=1"))
    assert n == 0
    assert llm.calls == []


def test_gap_proposal_registers_hypothesis_from_world_model(mem, tmp_path):
    from agentics.graph.nodes import _propose_gap_hypothesis

    mem.world_add("a01", "gap", "no data on tail latency under churn",
                  source="h#3")
    gap_child = json.dumps({"hypotheses": [_child(
        claim="a churn-aware queue depth bound keeps VNF tail latency "
              "under 200ms during membership churn events")]})
    llm = FakeLLM(("ROLE: gap-proposer", gap_child),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    assert asyncio.run(_propose_gap_hypothesis(ctx, state)) is True
    rows = mem.list_hypotheses(limit=5)
    assert rows and rows[0]["status"] == "proposed"
    events = [e["event"] for e in mem.recent_journal(limit=5, agent="a02")]
    assert "gap_hypothesis_registered" in events


def test_gap_proposal_off_without_ideation(mem, tmp_path):
    from agentics.graph.nodes import _propose_gap_hypothesis

    llm = FakeLLM()
    ctx = _ctx(mem, tmp_path, llm)
    ctx.settings = SimpleNamespace(**{**vars(_S()), "ideation_enabled": False})
    assert asyncio.run(_propose_gap_hypothesis(ctx, {})) is False
    assert llm.calls == []


# ------------------------------------------- deep-dive integration path

PROTOCOL = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': 0.93}))\n"
)


def test_deep_dive_settle_writes_ledger_and_harvests(mem, tmp_path):
    """Full loop: proposed -> tested (supported) -> ledger row -> world note
    -> follow-up child registered (Kosmos closed loop)."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    hid = _hyp(mem, "the toy precision method reaches >=0.9 precision")
    protocol_reply = json.dumps({"hypotheses": [{
        "claim": "the toy precision method reaches >=0.9 precision",
        "metric": "precision", "op": ">=", "target": 0.9,
        "protocol": PROTOCOL}]})
    ideator = json.dumps({"hypotheses": [_child()]})
    llm = FakeLLM(
        ("pre-registered by the swarm", protocol_reply),
        ("ROLE: ideator", ideator),
        ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    assert asyncio.run(_test_proposed_hypothesis(ctx, state)) is True
    row = mem.get_hypothesis(hid)
    assert row["status"] == "supported"
    # autoresearch ledger: keep/discard row with the measured value
    ledger = mem.ledger_recent(limit=5)
    assert ledger[0]["hyp_id"] == hid
    assert ledger[0]["status"] == "supported"
    assert ledger[0]["measured"] == pytest.approx(0.93)
    assert ledger[0]["source"] == "deep_dive"
    # Kosmos closed loop: the settled row spawned a gated child
    kids = [r for r in mem.list_hypotheses(limit=5) if r["parents"] == [hid]]
    assert len(kids) == 1


def test_deep_dive_retry_then_retire(mem, tmp_path):
    """Inconclusive near-miss gets ONE retry; when the retry yields no
    parsable protocol the claim is retired (autoresearch give-up rule)."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    hid = _hyp(mem, "an LSTM forecast predicts VNF load five minutes ahead",
               status="inconclusive", code="raise SystemExit(1)", attempts=1)
    llm = FakeLLM(("pre-registered by the swarm", "no parsable protocol here"))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    assert asyncio.run(_test_proposed_hypothesis(ctx, state)) is True
    row = mem.get_hypothesis(hid)
    assert row["status"] == "refuted"
    assert "retired" in row["verdict_note"]
    assert row["attempts"] == 2
    assert mem.testable_hypotheses(limit=6, max_attempts=2) == []
    events = [e["event"] for e in mem.recent_journal(limit=5, agent="a02")]
    assert "hypothesis_retired" in events


def test_deep_dive_retry_prompt_carries_failure_note(mem, tmp_path):
    """The retry prompt must carry the previous failure note (autoresearch:
    fix the bug if trivial, change the approach if broken)."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    hid = _hyp(mem, "an LSTM forecast predicts VNF load five minutes ahead",
               status="inconclusive", code="x = 1", attempts=1,
               )
    with mem._lock, mem._conn:
        mem._conn.execute(
            "UPDATE hypotheses SET verdict_note='runner error:boom' "
            "WHERE id=?", (hid,))
    llm = FakeLLM(("pre-registered by the swarm",
                   json.dumps({"hypotheses": [{
                       "claim": "an LSTM forecast predicts VNF load five "
                                "minutes ahead",
                       "metric": "load_err", "op": "<", "target": 5.0,
                       "protocol": PROTOCOL}]})))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    asyncio.run(_test_proposed_hypothesis(ctx, state))
    verify_calls = [p for purpose, p in llm.calls if purpose == "verify"]
    assert any("runner error:boom" in p and "CORRECTED protocol" in p
               for p in verify_calls)
