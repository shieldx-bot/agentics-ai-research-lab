"""Swarm evaluation — trajectory & collaboration metrics from real traces.

Seeds a small but complete lab history (cycles, claims, MRs, P19 panels
with defense, findings, LLM ledger, a cycle error + recovery, snapshots at
two swarm sizes) and asserts each of the five metric groups measures what
the taxonomy claims — with honest None when data is missing.
"""
import json
import time
from pathlib import Path

from agentics.memory.shared_memory import SharedMemory
from agentics.peer_review import PeerPanel
from agentics.swarm_eval import (_distinct_domains, attach_scalability,
                                 evaluate_swarm, percent_rows, scalability_S,
                                 save_snapshot)


def _seed_lab() -> SharedMemory:
    mem = SharedMemory(Path(":memory:"))
    _seed_into(mem)
    return mem


def _seed_into(mem: SharedMemory) -> None:
    """Seed a complete lab history into an EXISTING SharedMemory."""
    # --- three agents ran cycles: two productive, one idle ------------------
    for agent, cycle, outcome in (("a01", 1, "merged"), ("a02", 2, "merged"),
                                  ("a03", 3, "no_change")):
        mem.heartbeat(agent, "coder", cycle, status="idle")
        mem.journal(agent, "node", {"node": "scan", "action": "scan"})
        mem.journal(agent, "cycle_done", {"cycle": cycle, "summary": "s",
                                          "outcome": outcome})
    # a04 errored once, then recovered
    mem.heartbeat("a04", "writer", 1, status="idle")
    mem.post_message("a04", "alert", "cycle error: LLM timeout")
    mem.journal("a04", "cycle_done", {"cycle": 2, "summary": "recovered",
                                      "outcome": "merged"})
    # coordination friction + breaker + retry recovery
    mem.journal("a01", "claim", {"task": 1, "branch": "agent/a01/t1"})
    mem.journal("a01", "handoff", {"task": 1, "branch": "agent/a01/t1"})
    mem.journal("a01", "release", {"task": 1, "why": "cycle ended without MR"})
    mem.journal("a02", "lock_conflict", {"path": "docs/x.md"})
    mem.journal("a02", "budget_breaker_open", {"agent": "a02"})
    mem.journal("a03", "plan_retry_recovered", {"mr": 9})
    mem.journal("orchestrator", "broadcast", "noise from machinery")
    mem.journal("owner", "manual_maintenance", {"swept": []})

    # --- backlog: one open + one closed task (G1 close rate) -----------------
    t1 = mem.create_task("task open", agent="seed")
    t2 = mem.create_task("task closed", agent="seed")
    mem.set_task_status(t2, "closed")
    # the MR flow below flips tasks 1/2 to in_review — add one truly merged
    t3 = mem.create_task("task merged", agent="seed")
    mem.set_task_status(t3, "merged")

    # --- hypotheses: one supported, one still proposed (G2 resolution) -------
    h1 = mem.add_hypothesis("a01", "claim one", {"metric": "m", "op": ">=",
                                                 "target": 1})
    mem.add_hypothesis("a02", "claim two pending", {"metric": "m",
                                                    "op": ">=", "target": 1})
    mem.update_hypothesis(h1, "supported")

    # --- two MRs with panels: one approved cross-agent, one rejected ---------
    engine = PeerPanel(mem, panel_size=2)
    mr_ok = mem.open_review(task_id=1, title="good", body="b", author="a01",
                            branch="agent/a01/t1", files=["a.py"])
    mem.ensure_panel(mr_ok)
    engine.cast_vote(mr_ok, "a02", "accept", "lgtm")
    engine.cast_vote(mr_ok, "a03", "accept", "lgtm")   # unanimous → approved
    mr_bad = mem.open_review(task_id=2, title="bad", body="b", author="a02",
                             branch="agent/a02/t2", files=["b.py"])
    mem.ensure_panel(mr_bad)
    engine.cast_vote(mr_bad, "a03", "accept", "fine")  # accept first...
    engine.cast_vote(mr_bad, "a01", "reject", "metric not measured")  # → defense
    engine.submit_defense(mr_bad, "a02", {"a01": "still not measured"}, note="…")
    engine.cast_vote(mr_bad, "a01", "reject", "still not measured")  # r2 → rejected

    # --- traces: merged outcomes by two distinct agents ---------------------
    mem.add_trace("a01", "coder", 1, "merged", issue=1, category="code",
                  pr_number=1, files=["a.py"])
    mem.add_trace("a04", "writer", 2, "merged", issue=2, category="docs",
                  pr_number=2, files=["d.md"])
    mem.add_trace("a02", "coder", 3, "failed", issue=3, category="code",
                  failure="boom", files=[])

    # --- findings with citations across three domains ------------------------
    mem.add_finding("a01", "f1", "content",
                    citations=["https://arxiv.org/abs/2306.08543",
                               "https://arxiv.org/html/2306.08543v3"])
    mem.add_finding("a04", "f2", "content",
                    citations=["https://proceedings.iclr.cc/paper/x",
                               "https://github.com/example/repo"])

    # --- LLM ledger ----------------------------------------------------------
    mem.record_llm_call("a01", "research", ok=True, duration_s=30.0)
    mem.record_llm_call("a01", "plan", ok=True, duration_s=60.0)
    mem.record_llm_call("a02", "review", ok=False, duration_s=5.0)


def test_distinct_domains_collapses_mirror_urls():
    assert _distinct_domains(["https://arxiv.org/abs/1",
                              "https://arxiv.org/html/1v3",
                              "https://www.github.com/a/b"]) == {
        "arxiv.org", "github.com"}
    assert _distinct_domains(["not a url at all"]) == set()


def test_coordination_and_task_group():
    mem = _seed_lab()
    r = evaluate_swarm(mem)
    c = r["groups"]["coordination_task"]
    # 2 merged of 3 judged traces
    assert abs(c["task_completion_rate"] - 2 / 3) < 1e-9
    # both decided panels had >= 2 distinct voters
    assert c["panels_decided"] == 2 and c["panels_with_cross_review"] == 2
    assert abs(c["coordination_effectiveness"] - 1.0) < 1e-9
    # citations span arxiv.org, proceedings.iclr.cc, github.com
    assert c["citation_domains"] == 3 and c["findings"] == 2
    assert abs(c["exploration_coverage"] - 3 / 4) < 1e-9
    # authors with terminal outcomes: a01, a04, a02 = 3 of 4 active agents
    assert r["agents_active"] == 4
    assert abs(c["solution_diversity"] - 3 / 4) < 1e-9
    # backlog: 1 merged/closed of 3 tasks (2 sit in_review from the panels)
    assert abs(c["task_close_rate"] - 1 / 3) < 1e-9 and c["tasks_open"] == 1
    # balance = 1 - busiest-agent share of non-system journal events
    assert 0.0 <= c["workload_balance"] < 1.0


def test_convergence_dynamics_group():
    mem = _seed_lab()
    d = evaluate_swarm(mem)["groups"]["convergence_dynamics"]
    assert abs(d["convergence_correctness"] - 0.5) < 1e-9   # 1 of 2 approved
    # approved closed in round 1; rejected went through the defense (round 2)
    assert d["mean_panel_rounds"] == 1.5
    # one hypothesis decided (supported) of two registered
    assert abs(d["hypothesis_resolution_rate"] - 0.5) < 1e-9
    assert d["hypotheses_total"] == 2


def test_trajectory_and_handoff_group():
    mem = _seed_lab()
    h = evaluate_swarm(mem)["groups"]["trajectory_handoff"]
    # a01/a02/a04 produced outcomes; a03's cycle was a no_change loop
    assert h["total_cycles"] == 4 and h["productive_cycles"] == 3
    assert abs(h["productive_vs_looping"] - 3 / 4) < 1e-9
    assert h["trajectory_length_nodes"] == 3 / 4            # 3 node events / 4 cycles
    # handoff churn (release/claim) was removed: tautological in this codebase
    assert "handoff_churn" not in h
    dur = h["mean_cycle_duration_s"]
    assert dur is None or dur >= 0
    assert h["claim_stability"] == 0.0                      # 2 claims, 0 reconfirms
    assert h["claim_friction_events"] == 1


def test_resilience_group_and_cost():
    mem = _seed_lab()
    r = evaluate_swarm(mem)
    res = r["groups"]["resilience_scalability"]
    assert abs(res["llm_ok_rate"] - 2 / 3) < 1e-9
    assert res["budget_breaker_opens"] == 1
    # a04 errored once and its next cycle_done exists → full recovery
    assert res["error_recovery_rate"] == 1.0
    assert res["plan_retry_recovered"] == 1
    # no evidence rows yet -> honest None; no role-prompt feedback either
    assert res["evidence_pass_rate"] is None
    assert res["role_prompt_success_rate"] is None
    cost = r["groups"]["governance_cost"]
    assert cost["llm_calls"] == 3
    assert cost["llm_purpose_spread"] == 3                    # research/plan/review
    assert abs(cost["llm_top_purpose_share"] - 1 / 3) < 1e-9
    assert abs(cost["llm_mean_duration_s"] - 95 / 3) < 1e-9
    assert abs(cost["llm_calls_per_merged"] - 3 / 2) < 1e-9
    assert abs(cost["llm_seconds_per_merged"] - 95 / 2) < 1e-9


def test_incoherence_and_quality_gap():
    mem = _seed_lab()
    v = evaluate_swarm(mem)["groups"]["governance_cost"]
    # rejected panel's objection stood after defense → objector_unmoved = 1
    assert v["rejected_why"].get("objector_unmoved") == 1
    assert abs(v["incoherence_rate"] - 1 / 2) < 1e-9
    # rejected panel: 1 objector (a01); approved: 0 → gap = 1
    assert v["quality_gap_objectors"] == 1.0


def test_scalability_elasticity_across_sizes():
    mem = _seed_lab()
    r1 = evaluate_swarm(mem, now=time.time())
    r1["agents_active"] = 2
    r1["groups"]["coordination_task"]["task_completion_rate"] = 0.5
    save_snapshot(mem, r1)
    r2 = evaluate_swarm(mem, now=time.time())
    r2["agents_active"] = 4
    r2["groups"]["coordination_task"]["task_completion_rate"] = 2 / 3
    attach_scalability(mem, r2)
    res = r2["groups"]["resilience_scalability"]
    # %ΔP = (0.666-0.5)/0.5 = 1/3 ; %ΔN = (4-2)/2 = 1 → S = 1/3
    assert res["scalability_S"] is not None
    assert abs(res["scalability_S"] - 1 / 3) < 1e-9


def test_scalability_needs_different_size():
    mem = _seed_lab()
    r = evaluate_swarm(mem)
    assert scalability_S([], r)["S"] is None
    same = {"agent_count": r["agents_active"],
            "report": {"groups": {"coordination_task":
                                  {"task_completion_rate": 0.9}}}}
    assert scalability_S([same], r)["S"] is None
    other = {"agent_count": r["agents_active"] + 1, "report": {"groups": {
        "coordination_task": {"task_completion_rate": 0.6}}}}
    s = scalability_S([same, other], r)["S"]
    assert s is not None


def test_empty_lab_reports_honest_nones():
    mem = SharedMemory(Path(":memory:"))
    r = evaluate_swarm(mem)
    assert r["agents_active"] == 0
    g = r["groups"]
    assert g["coordination_task"]["task_completion_rate"] is None
    assert g["coordination_task"]["task_close_rate"] is None
    assert g["coordination_task"]["workload_balance"] is None
    assert g["trajectory_handoff"]["productive_vs_looping"] is None
    assert g["trajectory_handoff"]["mean_cycle_duration_s"] is None
    assert g["governance_cost"]["incoherence_rate"] is None
    assert g["resilience_scalability"]["llm_ok_rate"] is None
    assert g["resilience_scalability"]["evidence_pass_rate"] is None
    assert g["resilience_scalability"]["role_prompt_success_rate"] is None
    assert g["convergence_dynamics"]["hypothesis_resolution_rate"] is None


def test_evidence_and_role_prompt_rates():
    mem = _seed_lab()
    ev = mem.record_evidence("a01", task_id=1, cmd="true", exit_code=0,
                             verdict="confirmed", hypothesis="h",
                             method="m", output="ok")
    mem.record_evidence("a01", task_id=1, cmd="false", exit_code=1,
                        verdict="refuted", hypothesis="h", method="m",
                        output="bad")
    mem.store_role_prompt("coder", "code", mission="do the thing")
    mem.bump_role_prompt("coder", "code", success=True)
    mem.bump_role_prompt("coder", "code", success=False)
    r = evaluate_swarm(mem)
    res = r["groups"]["resilience_scalability"]
    assert res["evidence_total"] == 2
    assert res["evidence_pass_rate"] == 0.5
    assert res["role_prompt_success_rate"] == 0.5


def test_report_is_json_serializable_and_snapshots_roundtrip():
    mem = _seed_lab()
    report = evaluate_swarm(mem)
    blob = json.dumps(report, ensure_ascii=False)   # must not raise
    assert "groups" in blob
    sid = save_snapshot(mem, report)
    snaps = mem.list_eval_snapshots(limit=5)
    assert snaps and snaps[0]["id"] == sid
    assert snaps[0]["agent_count"] == report["agents_active"]
    assert snaps[0]["report"]["groups"] == report["groups"]


def test_percent_rows_map_to_percent_scale():
    mem = _seed_lab()
    rows = percent_rows(evaluate_swarm(mem))
    by = {(r["group"], r["metric"]): r for r in rows}
    # rate -> direct x100 (2 merged of 3 judged)
    tc = by[(1, "Tỷ lệ hoàn thành nhiệm vụ")]
    assert tc["pct"] == 66.7 and tc["raw"] == "2/3"
    # mean panel rounds against the P19 cap of 2 (1.5/2)
    assert by[(2, "Vòng vote trung bình (tối đa 2)")]["pct"] == 75.0
    # rejected_why share of decided panels (1 objector_unmoved of 2)
    why = by[(5, "Lý do reject: objector_unmoved")]
    assert why["pct"] == 50.0 and why["raw"] == "1/2"
    # no replications decided yet -> honest None, never a faked %
    assert by[(4, "Replication kiểm chứng (P24)")]["pct"] is None
    # count without a natural denominator keeps '—'
    assert by[(1, "Số nhóm việc (category)")]["pct"] is None
    # new metrics land on the same percent scale (1 supported of 2 hypotheses)
    assert by[(2, "Giả thuyết đã quyết (P24)")]["pct"] == 50.0
    assert by[(1, "Tỷ lệ đóng nhiệm vụ (backlog)")]["pct"] == 33.3
    assert by[(4, "Evidence thông kiểm chứng")]["pct"] is None
    # churn row is gone, replaced by duration + claim stability
    assert not any("churn" in r["metric"] for r in rows)
    # every row carries a reading guide and a health direction
    assert all(r["hint"] and isinstance(r["good_high"], bool) for r in rows)


def test_percent_rows_survive_empty_lab():
    rows = percent_rows(evaluate_swarm(SharedMemory(Path(":memory:"))))
    assert rows
    # nothing measurable -> only the coverage/diversity guards yield 0.0,
    # every other metric keeps the honest '—'
    assert all(r["pct"] in (None, 0.0) for r in rows)


def test_system_seeders_are_not_swarm_members():
    mem = SharedMemory(Path(":memory:"))
    mem.heartbeat("a01", "coder", 1, status="idle")
    mem.journal("a01", "cycle_done",
                {"cycle": 1, "summary": "s", "outcome": "merged"})
    mem.journal("seed", "task_created", {"task": 1})
    mem.journal("supervisor", "backlog_top_up", {"task": 2})
    r = evaluate_swarm(mem)
    assert r["agents_active"] == 1
    assert r["agents_active_names"] == ["a01"]


def test_cli_swarm_eval_percent_view(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    mem = SharedMemory(s.db_path)
    _seed_into(mem)
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(
        cli.app, ["swarm-eval", "--percent", "--no-save"])
    assert result.exit_code == 0, result.output
    assert "Bảng % tổng hợp" in result.output
    assert "67%" in result.output          # 2/3 completion
    assert "75%" in result.output          # 1.5 panel rounds / 2


def test_cli_swarm_eval_renders_and_saves(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    mem = SharedMemory(s.db_path)
    _seed_into(mem)
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["swarm-eval"])
    assert result.exit_code == 0, result.output
    for header in ("Coordination & Task", "Convergence", "Trajectory",
                   "Resilience", "Governance"):
        assert header in result.output
    assert "snapshot #" in result.output
    mem2 = SharedMemory(s.db_path)
    snaps = mem2.list_eval_snapshots()
    mem2.close()
    assert len(snaps) == 1 and snaps[0]["agent_count"] == 4
