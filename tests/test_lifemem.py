"""LifeMem memory upgrade — clustering, MRL splitting, distillation, recall."""

import math

from agentics.memory import lifemem
from agentics.memory.lifemem import (
    assign_trace,
    distill_skill,
    embed,
    maybe_split,
    mean_resultant_length,
    on_merge,
    skill_context,
    two_means_split,
)
from agentics.memory.shared_memory import SharedMemory


# ---------------------------------------------------------------- geometry
def test_embed_is_deterministic_unit_norm_and_discriminative():
    a1 = embed("parse yaml config file")
    a2 = embed("parse yaml config file")
    b = embed("merge pull request on github")
    assert a1 == a2
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)
    assert lifemem.cosine(a1, b) < 0.3          # different workflows differ
    assert embed("") == [0.0] * 64              # no words -> zero vector


def test_mrl_bounds():
    v = embed("run pytest and push branch")
    identical = [v] * 7                          # perfectly redundant cluster
    assert mean_resultant_length(identical) > 0.99
    spread = [embed(t) for t in (
        "merge pull request on github",
        "write documentation prose for readme",
        "query database index performance",
        "parse yaml configuration file",
        "render chart with matplotlib library",
        "deploy docker container to cluster",
        "review source code diff comments",
        "schedule cron job for backups",
        "compile rust crate with cargo",
        "translate sentences to french language",
    )]
    assert mean_resultant_length(spread) < 0.5   # heterogeneous cluster


def test_two_means_split_separates_two_groups_and_is_deterministic():
    vecs = [embed(t) for t in (
        "pytest suite runner assertions", "pytest suite runner fixtures",
        "pytest suite runner coverage",  "pytest suite runner mocks",
        "readme documentation prose",    "readme documentation prose guide",
        "readme documentation prose intro", "readme documentation prose faq",
    )]
    labels1 = two_means_split(vecs)
    labels2 = two_means_split(vecs)
    assert labels1 == labels2
    group0 = {i for i, lb in enumerate(labels1) if lb == 0}
    group1 = {i for i, lb in enumerate(labels1) if lb == 1}
    assert len(group0) == 4 and len(group1) == 4
    assert group0.isdisjoint(group1)


# ---------------------------------------------------------------- clustering
def test_similar_traces_join_the_same_cluster_different_open_new(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        t1 = m.add_trace("a1", "coder", 1, "pending", category="code",
                         attempted="add pytest fixtures for claim tests")
        t2 = m.add_trace("a2", "coder", 2, "pending", category="code",
                         attempted="add pytest fixtures for claim module")
        t3 = m.add_trace("a3", "coder", 3, "pending", category="code",
                         attempted="rewrite readme deployment section")
        c1 = assign_trace(m, t1, "code", "add pytest fixtures for claim tests")
        c2 = assign_trace(m, t2, "code", "add pytest fixtures for claim module")
        c3 = assign_trace(m, t3, "code", "rewrite readme deployment section")
        assert c1 == c2                     # similar workflow -> same cluster
        assert c3 != c1                     # different workflow -> new cluster
        # idempotent: already-clustered traces are never re-filed
        assert assign_trace(m, t1, "code", "totally new text") == c1
    finally:
        m.close()


def test_clusters_are_scoped_by_category(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        t1 = m.add_trace("a1", "coder", 1, "pending", category="code",
                         attempted="add pytest fixtures for claim tests")
        t2 = m.add_trace("a2", "coder", 2, "pending", category="docs",
                         attempted="add pytest fixtures for claim tests")
        c1 = assign_trace(m, t1, "code", "add pytest fixtures for claim tests")
        c2 = assign_trace(m, t2, "docs", "add pytest fixtures for claim tests")
        assert c1 != c2
        assert m.get_cluster(c1)["category"] == "code"
        assert m.get_cluster(c2)["category"] == "docs"
    finally:
        m.close()


def test_empty_text_trace_is_not_clustered(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        t = m.add_trace("a1", "coder", 1, "yield", category="other", attempted="")
        assert assign_trace(m, t, "other", "") is None
        assert m.trace_cluster(t) is None
    finally:
        m.close()


# ---------------------------------------------------------------- splitting
def test_maybe_split_fires_only_on_large_and_loose_clusters(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "mixed bag")
        # two very different workflow families interleaved in ONE cluster
        texts_a = ("pytest suite runner assertions", "pytest suite runner fixtures",
                   "pytest suite runner coverage",  "pytest suite runner mocks")
        texts_b = ("readme documentation prose", "readme documentation prose guide",
                   "readme documentation prose intro", "readme documentation prose faq")
        for t_text in texts_a:
            t = m.add_trace("a1", "coder", 1, "pending", category="code",
                            attempted=t_text)
            m.set_trace_cluster(t, cid)
        for t_text in texts_b:
            t = m.add_trace("a2", "coder", 1, "pending", category="code",
                            attempted=t_text)
            m.set_trace_cluster(t, cid)
        assert maybe_split(m, cid, volume=6, tau_mrl=0.95) is True
        texts = [mm["attempted"] for mm in m.cluster_members(cid)]
        assert len(texts) == 4 and all("pytest" in t for t in texts)
        # a large but homogeneous cluster must NOT split
        cid2 = m.create_cluster("code", "samey")
        for i in range(8):
            t = m.add_trace("a1", "coder", i, "pending", category="code",
                            attempted="identical retry loop implementation")
            m.set_trace_cluster(t, cid2)
        assert maybe_split(m, cid2, volume=6, tau_mrl=0.95) is False
        # a loose but small cluster must NOT split either
        cid3 = m.create_cluster("code", "tiny")
        for i in range(3):
            t = m.add_trace("a1", "coder", i, "pending", category="code",
                            attempted=f"totally different task {i}")
            m.set_trace_cluster(t, cid3)
        assert maybe_split(m, cid3, volume=6, tau_mrl=0.95) is False
    finally:
        m.close()


def test_maybe_split_keeps_labels_aligned_when_empty_trace_is_present(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "mixed workflows")
        texts_a = ("pytest suite runner assertions", "pytest suite runner fixtures",
                   "pytest suite runner coverage", "pytest suite runner mocks")
        texts_b = ("readme documentation prose", "readme documentation prose guide",
                   "readme documentation prose intro", "readme documentation prose faq")
        for i, value in enumerate(texts_a):
            t = m.add_trace("a1", "coder", i, "merged", category="code", attempted=value)
            m.set_trace_cluster(t, cid)
        empty = m.add_trace("a1", "coder", 99, "pending", category="code", attempted="")
        m.set_trace_cluster(empty, cid)
        for i, value in enumerate(texts_b):
            t = m.add_trace("a2", "coder", i, "merged", category="code", attempted=value)
            m.set_trace_cluster(t, cid)

        assert maybe_split(m, cid, volume=6, tau_mrl=0.95) is True
        original = m.cluster_members(cid)
        moved = [row for row in m.cluster_members(m.list_clusters("code")[-1]["cluster_id"])
                 if row["attempted"]]
        assert {row["attempted"] for row in original if row["attempted"]} == set(texts_a)
        assert {row["attempted"] for row in moved} == set(texts_b)
        assert m.trace_cluster(empty) == cid
        # Each resulting cluster receives a skill distilled only from its merged traces.
        assert len(m.list_skills("code")) == 2
    finally:
        m.close()


# ---------------------------------------------------------------- distillation
def test_distill_requires_verified_merges(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "unverified")
        t = m.add_trace("a1", "coder", 1, "pending", category="code",
                        attempted="add retry loop", pr_number=5)
        m.set_trace_cluster(t, cid)
        assert distill_skill(m, cid) is None        # nothing merged yet
    finally:
        m.close()


def test_heuristic_distillation_captures_workflow_and_failure(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "retry loop")
        for i in range(2):
            t = m.add_trace("a1", "coder", i, "merged", category="code",
                            attempted="add pytest fixtures then push branch",
                            pr_number=10 + i, files=["tests/test_x.py"])
            m.set_trace_cluster(t, cid)
        tf = m.add_trace("a2", "coder", 3, "failed", category="code",
                         attempted="add pytest fixtures", failure="flake8 lint errors")
        m.set_trace_cluster(tf, cid)
        skill = distill_skill(m, cid)
        assert skill and "Workflow (code)" in skill
        assert "Avoid: flake8 lint errors" in skill
        skills = m.list_skills("code")
        assert len(skills) == 1 and skills[0]["cluster_id"] == cid
    finally:
        m.close()


def test_distill_supersedes_previous_skill_and_llm_path(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "retry loop")
        t = m.add_trace("a1", "coder", 1, "merged", category="code",
                        attempted="add retry loop", pr_number=1)
        m.set_trace_cluster(t, cid)
        distill_skill(m, cid)
        llm_seen = {}

        def fake_llm(prompt: str) -> str:
            llm_seen["prompt"] = prompt
            return "1. Run the suite. 2. Rebase. Avoid: pushing without tests."

        assert distill_skill(m, cid, llm_ask=fake_llm)
        assert "successful runs" in llm_seen["prompt"]
        active = m.list_skills("code")
        assert len(active) == 1 and "Rebase" in active[0]["text"]
        st = m.memory_stats()
        assert st["superseded_skills"] == 1 and st["active_skills"] == 1
    finally:
        m.close()


def test_on_merge_clusters_and_distills_end_to_end(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        t = m.add_trace("a1", "coder", 1, "pending", category="code",
                        attempted="add pytest fixtures for claim tests", pr_number=7)
        m.set_trace_outcome(7, "merged")            # real flow: outcome flips first
        touched = on_merge(m, [t])
        assert touched and m.get_trace(t)["cluster_id"] == touched[0]
        skills = m.list_skills("code")
        assert len(skills) == 1
        # second merge of a similar workflow refreshes the same cluster's skill
        t2 = m.add_trace("a2", "coder", 2, "pending", category="code",
                         attempted="add pytest fixtures for claim module", pr_number=8)
        m.set_trace_outcome(8, "merged")
        touched2 = on_merge(m, [t2])
        assert touched2 == touched
        assert len(m.list_skills("code")) == 1      # still one active skill
    finally:
        m.close()


# ---------------------------------------------------------------- recall
def test_skill_context_ranks_by_keywords_and_bumps_uses(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    try:
        cid = m.create_cluster("code", "retry loop")
        t = m.add_trace("a1", "coder", 1, "merged", category="code",
                        attempted="add pytest fixtures then push branch", pr_number=1)
        m.set_trace_cluster(t, cid)
        distill_skill(m, cid)
        cid2 = m.create_cluster("docs", "docs workflow")
        t2 = m.add_trace("a2", "writer", 2, "merged", category="docs",
                         attempted="rewrite readme prose", pr_number=2)
        m.set_trace_cluster(t2, cid2)
        distill_skill(m, cid2)

        ctx = skill_context(m, "add pytest fixtures for tests", category="code")
        assert "pytest fixtures" in ctx
        assert "docs workflow" not in ctx
        code_skill = m.list_skills("code")[0]
        assert code_skill["uses"] == 1              # recall = activation
        empty = skill_context(m, "anything", category="experiment")
        assert empty == "(no distilled skills yet)"
    finally:
        m.close()


# ---------------------------------------------------------------- wiring
def test_memory_stats_and_migration_from_pre_lifemem_db(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)                      # simulate an old DB without cluster_id
    conn.execute("CREATE TABLE traces (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, "
                 "agent TEXT, role TEXT DEFAULT '', cycle INTEGER DEFAULT 0, issue INTEGER, "
                 "category TEXT DEFAULT 'other', pr_number INTEGER, outcome TEXT, "
                 "attempted TEXT DEFAULT '', failure TEXT DEFAULT '', files TEXT DEFAULT '[]')")
    conn.execute("INSERT INTO traces (ts, agent, outcome) VALUES (1, 'old', 'merged')")
    conn.commit()
    conn.close()
    m = SharedMemory(db)
    try:
        assert m.get_trace(1)["cluster_id"] is None  # migrated column exists
        st = m.memory_stats()
        assert st["traces"] == 1 and st["clusters"] == 0
    finally:
        m.close()
