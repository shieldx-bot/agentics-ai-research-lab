"""P16 — the shared evidence table: verdicts as common knowledge (H3).

Recording evidence must (a) store the structured run result, (b) broadcast it
so every agent sees it (messages + journal), and (c) stay queryable by task,
by recency and by term — an agent must be able to ask "what did the others
already verify about X?" and reuse it instead of re-asking an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentics.memory.shared_memory import SharedMemory


def _mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "m.db")


def test_record_and_read_roundtrip(tmp_path):
    m = _mem(tmp_path)
    ev_id = m.record_evidence(
        "a04", 7, "python -m pytest test_x.py", exit_code=0,
        verdict="confirmed", hypothesis="cache halves the API calls",
        duration_s=1.25, artifact="experiments/check_cache.py",
    )
    rows = m.evidence_for_task(7)
    assert len(rows) == 1 and rows[0]["id"] == ev_id
    row = rows[0]
    assert row["agent"] == "a04" and row["verdict"] == "confirmed"
    assert row["hypothesis"] == "cache halves the API calls"
    assert row["exit_code"] == 0 and row["duration_s"] == 1.25
    assert row["artifact"] == "experiments/check_cache.py"


def test_recording_broadcasts_to_every_agent(tmp_path):
    m = _mem(tmp_path)
    m.record_evidence("a02", 3, "python exp.py", 1, "refuted",
                      hypothesis="linear beats quadratic")
    msgs = m.recent_messages()
    ev_msgs = [x for x in msgs if x["kind"] == "evidence"]
    assert len(ev_msgs) == 1
    assert "REFUTED" in ev_msgs[0]["content"] and "#3" in ev_msgs[0]["content"]
    acts = [j for j in m.recent_journal(limit=10) if j.get("event") == "evidence"]
    assert acts
    assert json.loads(acts[-1]["detail"])["verdict"] == "refuted"


def test_output_is_bounded(tmp_path):
    m = _mem(tmp_path)
    m.record_evidence("a01", None, "python big.py", 0, "confirmed",
                      output="y" * 5000)
    assert len(m.evidence_for_task(None)[0]["output"]) == 1500


def test_reviewer_can_reverdict(tmp_path):
    m = _mem(tmp_path)
    ev_id = m.record_evidence("a04", 5, "python check.py", 0, "confirmed")
    assert m.set_evidence_verdict(
        ev_id, "inconclusive", "check asserts nothing — not evidence", by="a07")
    row = m.evidence_for_task(5)[0]
    assert row["verdict"] == "inconclusive"
    assert row["verdict_by"] == "a07" and "not evidence" in row["note"]
    assert not m.set_evidence_verdict(999, "confirmed", by="a07")


def test_search_and_counts_across_agents(tmp_path):
    m = _mem(tmp_path)
    m.record_evidence("a02", 1, "python -m pytest t_cache.py", 0, "confirmed",
                      hypothesis="scan cache reduces API pressure")
    m.record_evidence("a04", 2, "python -m pytest t_merge.py", 1, "refuted",
                      hypothesis="merge is always fast")
    m.record_evidence("a09", 3, "python flaky.py", 124, "inconclusive")
    hits = m.evidence_search("cache")
    assert len(hits) == 1 and hits[0]["task_id"] == 1
    assert m.evidence_counts() == {
        "confirmed": 1, "refuted": 1, "inconclusive": 1}
    feed = m.latest_evidence(limit=2)
    assert [r["task_id"] for r in feed] == [3, 2]     # newest first


def test_task_id_null_roundtrip(tmp_path):
    m = _mem(tmp_path)
    m.record_evidence("a01", None, "python smoke.py", 0, "confirmed")
    assert len(m.evidence_for_task(None)) == 1
