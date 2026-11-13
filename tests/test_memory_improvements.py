"""v2 memory improvements — closing the loops that were half-built.

Personal memory: dedup (hash identity, hits/last_seen), recency AND relevance
recall, bounded table, in-place migration of v1 databases.
Shared memory: messages/findings/decisions trim, get_finding, extended stats.
Graph wiring: _memory_context["lessons"], node_plan lesson/recall injection.
No network, no chromadb required (fakes stand in for the vector store)."""
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentics.memory.personal_memory as pm_mod
import agentics.memory.shared_memory as sm_mod
from agentics.graph import nodes
from agentics.memory.personal_memory import PersonalMemory, lesson_hash
from agentics.memory.shared_memory import SharedMemory


# ------------------------------------------------------------ personal memory
def test_lesson_dedup_bumps_hits(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    lid = pm.add_lesson("no LLM account available", issue=3)
    # same lesson, reworded (case + spacing) → dedup, not a second row
    pm.add_lesson("No   LLM  account   available", issue=4)
    st = pm.lesson_stats()
    assert st["lessons"] == 1
    assert st["recurrences"] == 2
    assert st["recurring"] == 1
    lessons = pm.recent_lessons()
    assert len(lessons) == 1 and lessons[0].endswith("(x2)")
    with pm._lock:
        rows = pm._conn.execute(
            "SELECT id, issue FROM lessons WHERE lesson_hash=?",
            (lesson_hash("no llm account available"),),
        ).fetchall()
    assert rows and rows[0][0] == lid and rows[0][1] == 4   # id stable, issue refreshed
    pm.close()


def test_lesson_recency_order_newest_first(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    pm.add_lesson("first lesson")
    time.sleep(0.01)
    pm.add_lesson("second lesson")
    time.sleep(0.01)
    pm.add_lesson("third lesson")
    lessons = pm.recent_lessons(limit=2)
    assert [l.split()[0] for l in lessons] == ["third", "second"]
    pm.close()


def test_relevant_lessons_rank_by_overlap_then_hits(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    pm.add_lesson("sqlite wal lock contention slows the dashboard")
    pm.add_lesson("sqlite wal lock contention slows the dashboard")   # hits=2
    pm.add_lesson("chroma embedding index grows too large")
    out = pm.relevant_lessons("how to fix sqlite lock timeouts", limit=2)
    assert out and "sqlite" in out[0] and "(x2)" in out[0]
    assert "chroma" in out[1]
    # without overlap info the ranking still returns lessons, hits first
    assert "sqlite" in pm.relevant_lessons("unrelated query")[0]
    pm.close()


def test_lesson_trim_keeps_newest(tmp_path: Path, monkeypatch):
    # TRIM_EVERY=1 → deterministic bound; production (50/500) allows the same
    # KEEP + EVERY - 1 window between trims the journal/evidence feeds have.
    monkeypatch.setattr(pm_mod, "_LESSONS_TRIM_EVERY", 1)
    monkeypatch.setattr(pm_mod, "_LESSONS_KEEP_ROWS", 3)
    pm = PersonalMemory("a01", tmp_path)
    for i in range(7):          # 7 distinct lessons → trim on every insert
        pm.add_lesson(f"lesson number {i}")
        time.sleep(0.002)
    st = pm.lesson_stats()
    assert st["lessons"] == 3
    kept = pm.recent_lessons(limit=10)
    assert kept and "lesson number 6" in kept[0]      # newest survived
    # a deduped lesson never counts as a new insert → no extra trim pressure
    pm.add_lesson("lesson number 6")
    assert pm.lesson_stats()["lessons"] == 3
    pm.close()


def test_migration_from_v1_db_with_duplicates(tmp_path: Path):
    # hand-build a v1 (pre-dedup) notebook exactly as the old schema wrote it
    db = tmp_path / "a01.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE lessons (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "issue INTEGER, lesson TEXT NOT NULL);"
        "CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL);"
    )
    now = time.time()
    conn.executemany(
        "INSERT INTO lessons (ts, issue, lesson) VALUES (?,?,?)",
        [(now - 3, 1, "no LLM account available"),
         (now - 2, 2, "No LLM account  available"),     # duplicate, reworded
         (now - 1, 3, "always run verify before opening an MR")],
    )
    conn.commit()
    conn.close()

    pm = PersonalMemory("a01", tmp_path)     # opens + migrates in place
    st = pm.lesson_stats()
    assert st["lessons"] == 2                 # duplicates collapsed
    assert st["recurrences"] == 3             # 2 pre-existing rows + hits sum
    assert st["recurring"] == 1
    lessons = pm.recent_lessons()
    assert any("(x2)" in l and "LLM" in l for l in lessons)
    pm.close()

    # re-open: migration is idempotent, hashes do not regress
    pm = PersonalMemory("a01", tmp_path)
    assert pm.lesson_stats()["lessons"] == 2
    pm.close()


def test_state_roundtrip_survives_upgrade(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    pm.set_state("last_task", "42")
    assert pm.get_state("last_task") == "42"
    assert pm.get_state("missing", "dflt") == "dflt"
    pm.close()


# ------------------------------------------------------------- shared memory
def test_get_finding(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    fid = mem.add_finding("a01", "Vector DB notes", "chroma content")
    f = mem.get_finding(fid)
    assert f and f["title"] == "Vector DB notes" and f["citations"] == []
    assert mem.get_finding(fid + 999) is None
    mem.close()


def test_message_trim(tmp_path: Path, monkeypatch):
    # TRIM_EVERY=2 → trims at inserts 2,4,6,8 → exactly KEEP rows remain.
    monkeypatch.setattr(sm_mod, "_MESSAGES_TRIM_EVERY", 2)
    monkeypatch.setattr(sm_mod, "_MESSAGES_KEEP_ROWS", 5)
    mem = SharedMemory(tmp_path / "m.db")
    for i in range(8):
        mem.post_message("a01", "broadcast", f"msg {i}")
    with mem._lock:
        n = mem._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 5
    assert mem.recent_messages(limit=1)[0]["content"] == "msg 7"
    mem.close()


def test_findings_and_decisions_trim(tmp_path: Path, monkeypatch):
    # every 2nd insert trims → after 6 inserts each table sits at its KEEP bound
    monkeypatch.setattr(sm_mod, "_FINDINGS_TRIM_EVERY", 2)
    monkeypatch.setattr(sm_mod, "_FINDINGS_KEEP_ROWS", 3)
    monkeypatch.setattr(sm_mod, "_DECISIONS_TRIM_EVERY", 2)
    monkeypatch.setattr(sm_mod, "_DECISIONS_KEEP_ROWS", 3)
    mem = SharedMemory(tmp_path / "m.db")
    newest = 0
    for i in range(6):
        newest = mem.add_finding("a01", f"t{i}", "body")
        mem.add_decision("a01", f"d{i}")
    st = mem.memory_stats()
    assert st["findings"] == 3 and st["decisions"] == 3
    # the newest finding survived and is still fetchable (recall safety)
    assert mem.get_finding(newest) is not None
    mem.close()


def test_memory_stats_extended_counts(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    mem.post_message("a01", "broadcast", "hi")
    mem.add_finding("a01", "t", "b")
    mem.add_decision("a01", "d")
    st = mem.memory_stats()
    expected = {"messages": 1, "findings": 1, "decisions": 1,
                "evidence": 0, "hypotheses": 0}
    for key, at_least in expected.items():
        assert key in st and st[key] >= at_least, f"{key}: {st.get(key)}"
    mem.close()


# ------------------------------------------------------------ graph wiring
class _FakeVector:
    def __init__(self, hits, available=True):
        self._hits, self.available = hits, available

    def search(self, query, n=5):
        return self._hits[:n]


def _ctx(mem: SharedMemory, vector, personal) -> SimpleNamespace:
    return SimpleNamespace(memory=mem, vector=vector, meta={"personal": personal})


def test_finding_recall_text_skips_ghost_rows(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    fid = mem.add_finding("a02", "WAL tuning", "sqlite WAL checkpoints keep the dashboard fast")
    hits = [
        {"text": "old", "meta": {"finding_id": "999"}, "distance": 0.1},   # trimmed/ghost
        {"text": "hit", "meta": {"finding_id": str(fid)}, "distance": 0.2},
        {"text": "no id", "meta": {"agent": "a02"}, "distance": 0.3},      # no finding_id
    ]
    out = nodes.finding_recall_text(_ctx(mem, _FakeVector(hits), None), "wal tuning")
    assert "WAL tuning" in out and f"#{fid}" in out and "a02" in out
    assert "ghost" not in out and len(out.splitlines()) == 1
    mem.close()


def test_finding_recall_text_disabled_and_empty(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    fid = mem.add_finding("a02", "T", "content here")
    hits = [{"text": "hit", "meta": {"finding_id": str(fid)}}]
    # chromadb unavailable → recall degrades to ""
    assert nodes.finding_recall_text(_ctx(mem, _FakeVector(hits, available=False), None), "q") == ""
    # empty query → ""
    assert nodes.finding_recall_text(_ctx(mem, _FakeVector(hits), None), "   ") == ""
    # search raising → never crashes
    class _Boom(_FakeVector):
        def search(self, query, n=5):
            raise RuntimeError("chroma gone")

    assert nodes.finding_recall_text(_ctx(mem, _Boom([], available=True), None), "q") == ""
    mem.close()


def test_lessons_text_prefers_relevance(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    pm.add_lesson("sqlite lock contention")
    pm.add_lesson("chroma index growth")
    out = nodes._lessons_text(
        _ctx(None, None, pm), "sqlite lock issue", limit=1)
    assert "sqlite" in out and "chroma" not in out
    # recency mode without a query
    assert "sqlite" in nodes._lessons_text(_ctx(None, None, pm), "", limit=2)
    pm.close()


def test_lessons_text_degrades_for_old_fakes():
    # pre-v2 test doubles only implement add_lesson → no crash, empty text
    assert nodes._lessons_text(_ctx(None, None, SimpleNamespace(add_lesson=lambda *a: None)), "q") == ""
    assert nodes._lessons_text(_ctx(None, None, SimpleNamespace()), "") == ""
    assert nodes._lessons_text(_ctx(None, None, None), "q") == ""
    # raising recall → "" (memory must never break a cycle)
    class _Boom:
        def relevant_lessons(self, q, limit=5):
            raise RuntimeError("db busy")

    assert nodes._lessons_text(_ctx(None, None, _Boom()), "q") == ""


def test_memory_context_carries_personal_lessons(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    pm = PersonalMemory("a01", tmp_path / "personal")
    pm.add_lesson("verify before opening an MR")
    snap = nodes._memory_context(_ctx(mem, _FakeVector([], available=False), pm))
    assert "verify before opening an MR" in snap["lessons"]
    assert snap["verified"] == "(nothing verified yet)"
    pm.close()
    mem.close()


def test_memory_context_lessons_placeholder_for_empty(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    snap = nodes._memory_context(_ctx(mem, _FakeVector([], available=False), None))
    assert snap["lessons"] == "(none yet)"
    mem.close()


def test_plan_prompt_injects_lessons_and_recall(tmp_path: Path, monkeypatch):
    """The plan prompt now carries the planner's OWN lessons and the swarm's
    related findings — the two loops this upgrade closed."""
    import asyncio
    import json

    mem = SharedMemory(tmp_path / "m.db")
    pm = PersonalMemory("a01", tmp_path / "personal")
    pm.add_lesson("plan parsing failed when JSON is wrapped in prose")
    fid = mem.add_finding("a02", "JSON parsing study",
                          "model answers wrap JSON in markdown fences")
    hits = [{"text": "hit", "meta": {"finding_id": str(fid)}, "distance": 0.2}]
    seen: dict = {}

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        seen["prompt"] = prompt
        return json.dumps(
            {"files": [], "steps": [], "commit_message": "m",
             "verify_cmds": [], "hypotheses": []}), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    ctx = _ctx(mem, _FakeVector(hits), pm)
    ctx.agent_id, ctx.role = "a01", None
    ctx.settings = SimpleNamespace(research_topic="memory upgrades")
    ctx.heartbeat = lambda status="working": None
    ctx.report = lambda **kw: None
    ctx.local = None
    state = {
        "issue_number": 7, "issue_title": "parse plan JSON robustly",
        "issue_body": "extract the JSON block even inside markdown",
        "papers": [{"answer": ""}],
    }
    out = asyncio.run(nodes.node_plan(ctx, state))
    p = seen["prompt"]
    assert "personal memory" in p and "plan parsing failed" in p
    assert "relevance recall" in p and f"#{fid}" in p
    assert out.get("plan")  # cycle unaffected
    pm.close()
    mem.close()
