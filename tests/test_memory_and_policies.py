"""Unit tests for memory & policies — no network, no tokens required."""
from pathlib import Path

from agentics.memory.shared_memory import SharedMemory
from agentics.policies.ownership import OwnershipPolicy


def test_claim_atomic(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    assert mem.claim_issue(1, "a01", ttl_s=60) is True
    assert mem.claim_issue(1, "a02", ttl_s=60) is False      # held by a01
    assert mem.claim_issue(1, "a01", ttl_s=60) is True       # re-claim by owner ok
    mem.release_claim(1, "a01")
    assert mem.claim_issue(1, "a02", ttl_s=60) is True       # free after release
    mem.close()


def test_claim_ttl_expiry(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    assert mem.claim_issue(2, "a01", ttl_s=-1) is True       # already expired
    assert mem.claim_issue(2, "a02", ttl_s=60) is True       # expired -> stealable
    mem.close()


def test_findings_roundtrip(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    fid = mem.add_finding("a01", "t", "body", ["https://arxiv.org/abs/2401.0001"], issue=7)
    items = mem.list_findings()
    assert items and items[-1]["id"] == fid and items[-1]["issue"] == 7
    assert items[-1]["citations"] == ["https://arxiv.org/abs/2401.0001"]
    mem.close()


def test_heartbeat_active_agents(tmp_path: Path):
    mem = SharedMemory(tmp_path / "m.db")
    mem.heartbeat("a01", "coder", 3, "idle")
    agents = mem.active_agents(stale_after_s=300)
    assert agents and agents[0]["agent"] == "a01"
    assert mem.active_agents(stale_after_s=-1) == []          # stale immediately
    mem.close()


def test_ownership_matrix():
    pol = OwnershipPolicy({
        "coder": ["agentics/", "tests/"],
        "writer": ["docs/", "README*"],
    })
    assert pol.owner_of("agentics/llm/pool.py") == "coder"
    assert pol.owner_of("docs/POLICIES.md") == "writer"
    assert pol.owner_of("random/new.py") is None
    # own territory -> ok
    ok, foreign = pol.evaluate("coder", ["agentics/x.py", "tests/t.py"], {"writer"})
    assert ok and not foreign
    # foreign files of an ACTIVE role -> yield
    ok, foreign = pol.evaluate(
        "coder", ["agentics/x.py", "docs/a.md", "docs/b.md"], {"writer"}
    )
    assert not ok and set(foreign) == {"docs/a.md", "docs/b.md"}
    # foreign files but role inactive -> allowed
    ok, foreign = pol.evaluate("coder", ["docs/a.md"], set())
    assert ok
