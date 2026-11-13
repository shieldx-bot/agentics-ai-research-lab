"""P16bis claim dedup — a SETTLED claim is reused, never re-run (H3).

Completes the sharing loop: verified knowledge is not only broadcast, it is
also ENFORCED — whichever agent meets the same hypothesis later (same claim,
reworded, other cycle, other agent) inherits the first machine verdict:
  * research path (node_verify): reuse entry, no new protocol run;
  * coder path (node_implement): a supported claim costs zero protocol runs;
    a refuted claim blocks the push exactly like a fresh refutation would —
    re-asserting a falsified claim cannot sneak past the gate.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory
from agentics.verify.hypothesis import claim_hash

from test_p16_bridge import BridgeWorkspace, _impl_ctx, _impl_state, _VALID_RAW


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


# ------------------------------------------------------------- memory level
def test_claim_hash_matches_reworded_claims():
    assert claim_hash("Cache  HIT-rate >= 50%!") == claim_hash("cache hit-rate >= 50%!")
    assert claim_hash("a") != claim_hash("b")


def test_settled_lookup_matches_variant_and_ignores_inconclusive(mem):
    crit = {"metric": "hit_rate", "op": ">=", "target": 0.5}
    hyp_id = mem.add_hypothesis("agent-a", "The cache HIT rate stays above 0.5",
                                crit, status="testing")
    # still 'testing' — not settled yet
    assert mem.settled_hypothesis_for_claim("the cache hit rate stays above 0.5") is None
    mem.update_hypothesis(hyp_id, "supported", evidence_id=3, note="measured 0.72")
    hit = mem.settled_hypothesis_for_claim("  The  CACHE  hit  rate  stays  above  0.5  ")
    assert hit is not None and hit["status"] == "supported"
    assert hit["id"] == hyp_id and hit["evidence_id"] == 3 and hit["agent"] == "agent-a"
    # a different claim never matches
    assert mem.settled_hypothesis_for_claim("somebody else's claim") is None
    # an inconclusive twin is NOT settled — it deserves a fresh run
    other = mem.add_hypothesis("agent-b", "another claim entirely", crit,
                               status="testing")
    mem.update_hypothesis(other, "inconclusive")
    assert mem.settled_hypothesis_for_claim("another claim entirely") is None


# ------------------------------------------------------------- research path
def _verify_spec(claim: str = "single-pass min/max equals the naive scan") -> dict:
    return {
        "claim": claim,
        "metric": "equivalence_rate",
        "op": ">=",
        "target": 0.9,
        "protocol": (
            "import random\n"
            "random.seed(0)\n"
            "data = [random.random() for _ in range(200)]\n"
            "lo = hi = data[0]\n"
            "for x in data:\n"
            "    if x < lo: lo = x\n"
            "    if x > hi: hi = x\n"
            "ok = (min(data), max(data)) == (lo, hi)\n"
            "print(f'AGENTICS_RESULT: {{\"equivalence_rate\": {1.0 if ok else 0.0}}}')\n"
        ),
    }


def _verify_ctx(mem, ws, agent_id="agent-b"):
    return SimpleNamespace(
        agent_id=agent_id, role=Role.RESEARCHER, memory=mem,
        workspace=ws, cycle=2, meta={},
        llm=SimpleNamespace(),        # truthy: the extraction call is faked below
        settings=SimpleNamespace(research_topic="NFV"),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
    )


def test_second_cycle_reuses_the_settled_verdict(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)

    def _set_llm_answer(spec):
        async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
            return json.dumps({"hypotheses": [spec]}), []
        monkeypatch.setattr(nodes, "_llm_ask", fake_llm)

    _set_llm_answer(_verify_spec())
    ctx = _verify_ctx(mem, ws)
    first = asyncio.run(nodes.node_verify(ctx, {
        "finding_id": 1, "finding_text": "finding text", "issue_number": 5}))
    rows = mem.evidence_for_task(5)
    assert len(rows) == 1 and rows[0]["verdict"] == "confirmed"
    assert [r["status"] for r in first["verifications"]] == ["supported"]
    assert first["verifications"][0].get("reused") is None   # really ran

    # a second agent/cycle meets the SAME claim (reworded). P24 composes:
    # agent-c FIRST independently replicates the settled claim (one agent's
    # verdict is not knowledge), THEN the dedup gate still reuses the
    # original verdict for the fresh-finding block — reuse, no re-run.
    _set_llm_answer(_verify_spec(claim="Single-pass  MIN/MAX equals the naive scan"))
    ctx_b = _verify_ctx(mem, ws, agent_id="agent-c")
    second = asyncio.run(nodes.node_verify(ctx_b, {
        "finding_id": 2, "finding_text": "finding text", "issue_number": 5}))
    assert "P24 replicated 1 claim(s)" in second["summary"]
    repl, v = second["verifications"][0], second["verifications"][1]
    assert repl["replication"] is True and repl["status"] == "confirmed"
    assert repl["evidence_id"] != rows[0]["id"]        # the replica's own row
    assert v["reused"] is True and v["status"] == "supported"
    # after the P23 replay the claim's evidence pointer moved to the LATEST
    # measurement (the independent replica's row) — reuse points there
    assert v["evidence_id"] == repl["evidence_id"]
    events = [j["event"] for j in mem.recent_journal(limit=15)]
    assert "hypothesis_reused" in events
    assert "hypothesis_replicated" in events and "claim_reconfirmed" in events


def test_a_genuinely_new_claim_still_runs(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    _set = [_verify_spec()]

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        return json.dumps({"hypotheses": [_set[0]]}), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    ctx = _verify_ctx(mem, ws)
    first = asyncio.run(nodes.node_verify(ctx, {
        "finding_id": 1, "finding_text": "t", "issue_number": 5}))
    assert first["verifications"][0]["status"] == "supported"
    # different claim from the same agent → a real second protocol run
    _set[0] = _verify_spec(claim="a completely different claim about kernels")
    second = asyncio.run(nodes.node_verify(ctx, {
        "finding_id": 1, "finding_text": "t", "issue_number": 5}))
    assert second["verifications"][0].get("reused") is None
    assert len(mem.evidence_for_task(5)) == 2


# ------------------------------------------------------------- coder path
def test_coder_path_reuses_supported_claim_without_rerunning(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    crit = {"metric": _VALID_RAW["metric"], "op": _VALID_RAW["comparison"],
            "target": _VALID_RAW["threshold"]}
    hyp_id = mem.add_hypothesis("agent-a", _VALID_RAW["claim"], crit,
                                status="testing")
    seed_ev = mem.record_evidence("agent-a", 1, "seed-protocol.py", 0,
                                  "confirmed", hypothesis=_VALID_RAW["claim"])
    mem.update_hypothesis(hyp_id, "supported", evidence_id=seed_ev,
                          note="measured hit_rate=0.72")

    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[dict(_VALID_RAW)])))
    assert ws.pushed                                   # supported reuse → push
    # no NEW protocol run (only the tier-1 '-m' checks), no NEW evidence row
    assert [c for c in ws.cmds if not (len(c) >= 3 and c[1] == "-m")] == []
    assert len(mem.evidence_for_task(1)) == 1
    assert any(j["event"] == "hypothesis_reused"
               for j in mem.recent_journal(limit=10))


def test_coder_path_refused_to_reassert_a_refuted_claim(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ws = BridgeWorkspace(tmp_path)
    crit = {"metric": _VALID_RAW["metric"], "op": _VALID_RAW["comparison"],
            "target": _VALID_RAW["threshold"]}
    hyp_id = mem.add_hypothesis("agent-a", _VALID_RAW["claim"], crit,
                                status="testing")
    mem.update_hypothesis(hyp_id, "refuted", evidence_id=1, note="measured 0.31")

    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(hypotheses=[dict(_VALID_RAW)])))
    assert not ws.pushed
    assert "ALREADY REFUTED" in state.get("yield_reason", "")
