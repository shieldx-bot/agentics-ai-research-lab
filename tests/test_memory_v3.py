"""Memory v3 — same ideas, closed loops (shared + personal memory hardening).

Everything here upgrades a mechanism that already existed:
- shared: the findings trim now carries into the vector tier (sync-trim),
  and the P7 research dedup gate falls back to SQLite keyword search when
  chromadb is offline (the fallback the policies layer already used);
- personal: lesson identity is number-blind (recurring failures embedding
  varying counts dedup to one (xN) row), and recall boosts lessons recorded
  on the same issue (the ``issue`` column finally read back).

No network, no chromadb required (fakes stand in for the vector store; the
one real-chroma test skips when the package is absent).
"""

import asyncio
import hashlib
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentics.memory.shared_memory as sm
from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.personal_memory import PersonalMemory, lesson_hash
from agentics.memory.shared_memory import SharedMemory
from agentics.memory.vector_store import VectorStore


# ======================================================================
# Personal memory — number-blind lesson identity (dedup, one notch stronger)
# ======================================================================
def test_lesson_hash_is_number_blind():
    """Varying counts/ids must not split one recurring failure into many."""
    assert lesson_hash("budget cap reached (57/50)") == lesson_hash("budget cap reached (63/50)")
    assert lesson_hash("PR #12 opened") == lesson_hash("PR #97 opened")
    # case/whitespace still normalized as before
    assert lesson_hash("No  LLM   account available") == lesson_hash("no llm account available")
    # genuinely different lessons stay different
    assert lesson_hash("tests failed") != lesson_hash("review rejected")


def test_recurring_failure_with_varying_counts_dedups_to_one_row(tmp_path: Path):
    pm = PersonalMemory("a01", tmp_path)
    pm.add_lesson("P18 budget: hourly pool cap reached (57/50 calls)", 7)
    pm.add_lesson("P18 budget: hourly pool cap reached (63/50 calls)", 7)
    pm.add_lesson("P18 budget: hourly pool cap reached (71/50 calls)", 8)
    stats = pm.lesson_stats()
    assert stats["lessons"] == 1          # one row, not three near-identical ones
    assert stats["recurrences"] == 3      # the pattern is visible as (x3)
    assert "(x3)" in pm.recent_lessons()[0]
    pm.close()


def test_v2_db_renormalizes_hashes_and_consolidates(tmp_path: Path):
    """A v2 notebook (digit-sensitive hashes, duplicate variants) upgrades in
    place: hashes recomputed, duplicates collapsed with hits rolled up."""
    db = tmp_path / "a02.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE lessons (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "issue INTEGER, lesson TEXT NOT NULL, lesson_hash TEXT NOT NULL DEFAULT '', "
        "hits INTEGER NOT NULL DEFAULT 1, last_seen REAL NOT NULL DEFAULT 0);"
        "CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "updated_at REAL NOT NULL);"
    )

    def v2_hash(text: str) -> str:   # the OLD (digit-sensitive) normalization
        return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:32]

    for text in ("cap reached (57/50)", "cap reached (63/50)", "unique lesson"):
        conn.execute(
            "INSERT INTO lessons (ts, lesson, lesson_hash, hits, last_seen) "
            "VALUES (?,?,?,?,?)", (time.time(), text, v2_hash(text), 1, time.time()),
        )
    conn.commit()
    conn.close()

    pm = PersonalMemory("a02", tmp_path)          # open = migrate
    stats = pm.lesson_stats()
    assert stats["lessons"] == 2                  # the two variants collapsed
    assert stats["recurrences"] == 3              # hits rolled up
    pm.close()

    pm2 = PersonalMemory("a02", tmp_path)         # re-open is idempotent
    assert pm2.lesson_stats() == stats
    pm2.close()


def test_relevant_lessons_issue_boost_reorders(tmp_path: Path):
    """Same-issue history outranks keyword overlap; without the issue the
    keyword ranking is unchanged (the boost is additive, not a rewrite)."""
    pm = PersonalMemory("a01", tmp_path)
    pm.add_lesson("keyword-rich unrelated lesson about docker network stack", 1)
    pm.add_lesson("totally different wording zero overlap", 42)
    assert "zero overlap" in pm.relevant_lessons(
        "neural network pruning", limit=2, issue=42)[0]
    assert "docker" in pm.relevant_lessons("neural network pruning", limit=2)[0]
    pm.close()


def test_lessons_text_passes_issue_and_degrades_for_old_fakes():
    """node-side: the issue reaches relevant_lessons, and a v2-shaped fake
    (no ``issue`` kwarg) still answers via the positional call."""

    class _V3:
        def relevant_lessons(self, query, limit=5, issue=None):
            _V3.seen = (query, limit, issue)
            return ["same-issue lesson"]

    class _V2:                                   # pre-v3 fake: no ``issue``
        def relevant_lessons(self, query, limit=5):
            return ["v2 lesson"]

    ctx = SimpleNamespace(meta={"personal": _V3()})
    out = nodes._lessons_text(ctx, "some query", issue=9)
    assert "same-issue lesson" in out and _V3.seen[2] == 9

    ctx_old = SimpleNamespace(meta={"personal": _V2()})
    assert "v2 lesson" in nodes._lessons_text(ctx_old, "some query", issue=9)


# ======================================================================
# Shared memory — vector tier trimmed in sync with the findings table
# ======================================================================
def test_vector_delete_many_and_count(tmp_path: Path):
    vs = VectorStore(tmp_path / "chroma")
    if not vs.available:
        pytest.skip("chromadb not installed")
    d1 = vs.add("alpha doc about sqlite wal tuning", {"finding_id": 1})
    d2 = vs.add("beta doc about chroma collections", {"finding_id": 2})
    assert d1 and d2 and vs.count() == 2
    assert vs.delete_many([d1, "", None]) == 1     # empty ids ignored
    assert vs.count() == 1
    assert vs.delete_many([]) == 0                 # nothing to do
    remaining = [h["meta"].get("finding_id") for h in vs.search("sqlite wal", n=1)]
    assert "1" not in remaining                    # the deleted doc is gone


def test_findings_trim_fires_vector_hook_with_evicted_docs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sm, "_FINDINGS_TRIM_EVERY", 2)
    monkeypatch.setattr(sm, "_FINDINGS_KEEP_ROWS", 1)
    mem = SharedMemory(tmp_path / "m.db")
    deleted: list[str] = []
    mem.set_vector_trim_hook(deleted.extend)

    f1 = mem.add_finding("a01", "first title", "content one")
    mem.set_finding_vector_doc(f1, "doc-1")
    assert deleted == []                          # trim not due at insert 1

    f2 = mem.add_finding("a01", "second title", "content two")
    mem.set_finding_vector_doc(f2, "doc-2")
    assert deleted == ["doc-1"]                   # exactly the evicted row's doc
    assert [r["id"] for r in mem.list_findings()] == [f2]
    assert mem.get_finding(f1) is None            # evicted row really gone
    mem.close()


def test_trim_hook_failure_never_breaks_the_insert(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sm, "_FINDINGS_TRIM_EVERY", 1)
    monkeypatch.setattr(sm, "_FINDINGS_KEEP_ROWS", 1)

    def boom(doc_ids):
        raise RuntimeError("chroma gone")

    mem = SharedMemory(tmp_path / "m.db")
    mem.set_vector_trim_hook(boom)
    f1 = mem.add_finding("a01", "t1", "c1")
    f2 = mem.add_finding("a01", "t2", "c2")       # trim fires, hook explodes
    assert [r["id"] for r in mem.list_findings()] == [f2]   # insert + trim survived
    assert mem.get_finding(f1) is None
    mem.close()


def test_set_finding_vector_doc_roundtrip(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    fid = mem.add_finding("a01", "title", "content")
    assert mem.get_finding(fid)["vector_doc_id"] == ""    # default: not linked
    mem.set_finding_vector_doc(fid, "doc-abc")
    assert mem.get_finding(fid)["vector_doc_id"] == "doc-abc"
    mem.set_finding_vector_doc(fid, "")                   # empty is a no-op
    assert mem.get_finding(fid)["vector_doc_id"] == "doc-abc"
    mem.close()


def test_finding_search_ranks_by_keyword_hits(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    mem.add_finding("a01", "Findings: sqlite WAL tuning", "c")
    mem.add_finding("a01", "Findings: sqlite WAL checkpoint tuning", "c")
    mem.add_finding("a01", "Findings: docker networking", "c")
    hits = mem.finding_search("sqlite WAL tuning", limit=5)
    assert hits and hits[0]["hits"] >= 3                   # best overlap first
    assert all("docker" not in h["title"] for h in hits)
    assert mem.finding_search("zzz", limit=5) == []        # no false positives
    mem.close()


# ======================================================================
# node_research — P7 dedup survives without chromadb (SQLite fallback)
# ======================================================================
class _OfflineVector:
    available = False

    def search(self, query, n=5):
        return []


def _research_ctx(mem):
    return SimpleNamespace(
        agent_id="a03", role=Role.RESEARCHER, memory=mem, vector=_OfflineVector(),
        llm=None, meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="agent memory",
                                 free_tools_enabled=False),
        heartbeat=lambda status="working": None, report=lambda **k: None,
    )


def test_research_dedup_falls_back_to_sqlite_when_vector_offline(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    for tag in ("alpha", "beta", "gamma"):
        mem.add_finding("a01", f"Findings: Survey NFV orchestration {tag}", "content")
    state = asyncio.run(nodes.node_research(_research_ctx(mem),
                                            {"issue_title": "Survey NFV orchestration patterns",
                                             "papers": [], "citations": []}))
    assert state["done"] is True
    assert "already covered" in state["summary"]           # the gate still fired
    assert len(mem.list_findings()) == 3                   # nothing new stored
    mem.close()


def test_research_dedup_fallback_ignores_weak_single_word_matches(tmp_path: Path):
    """One shared word is not coverage — only >= 2 significant-word overlap
    counts, so the fallback is not stricter or looser than intended."""
    mem = SharedMemory(tmp_path / "m.db")
    for tag in ("one", "two", "three"):
        mem.add_finding("a01", f"NFV notes {tag}", "content")
    state = asyncio.run(nodes.node_research(_research_ctx(mem),
                                            {"issue_title": "NFV orchestration patterns",
                                             "papers": [], "citations": []}))
    assert "already covered" not in state.get("summary", "")   # fell through
    assert state["error"] == "no LLM account available"        # hermetic degrade
    mem.close()


# ======================================================================
# _finish_research — the finding row is linked to its vector doc
# ======================================================================
class _LinkingVector:
    available = True

    def __init__(self):
        self.docs: list[tuple[str, dict]] = []

    def add(self, text, metadata=None):
        doc_id = f"doc-{len(self.docs) + 1}"
        self.docs.append((doc_id, metadata or {}))
        return doc_id

    def search(self, query, n=5):
        return []


def test_finish_research_links_vector_doc_to_finding(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    vector = _LinkingVector()
    ctx = SimpleNamespace(agent_id="a03", role=Role.RESEARCHER, memory=mem,
                          vector=vector, settings=SimpleNamespace(),
                          heartbeat=lambda status="working": None,
                          report=lambda **k: None)
    state = {"issue_number": 5, "papers": [], "citations": [], "bibtex": "",
             "issue_title": "T", "issue_body": ""}
    out = asyncio.run(nodes._finish_research(ctx, state, "agent memory",
                                             "the answer", ["https://arxiv.org/x"],
                                             "free"))
    fid = out["finding_id"]
    assert vector.docs[0][1]["finding_id"] == fid         # metadata carries the row id
    assert mem.get_finding(fid)["vector_doc_id"] == "doc-1"   # and the row carries the doc id
    mem.close()


def test_finish_research_survives_unlinkable_memory(tmp_path: Path):
    """An old fake memory without set_finding_vector_doc must not crash the
    research tail — linking is best-effort, recall degrades to ghost-skip."""

    class _OldMem(SharedMemory):
        set_finding_vector_doc = None                     # simulate pre-v3 surface

    mem = _OldMem(tmp_path / "m.db")
    ctx = SimpleNamespace(agent_id="a03", role=Role.RESEARCHER, memory=mem,
                          vector=_LinkingVector(), settings=SimpleNamespace(),
                          heartbeat=lambda status="working": None,
                          report=lambda **k: None)
    state = {"issue_number": 5, "papers": [], "citations": [], "bibtex": ""}
    out = asyncio.run(nodes._finish_research(ctx, state, "t", "a", [], "free"))
    assert out["finding_id"] > 0                          # the finding was stored
    mem.close()
