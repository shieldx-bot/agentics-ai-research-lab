"""Regression coverage for claim lease renewal, expiry, and cleanup."""

from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph.nodes import node_reflect
from agentics.memory.shared_memory import SharedMemory
from agentics.policies import claims as claims_module
from agentics.policies.claims import ClaimPolicy


@pytest.fixture
def memory(tmp_path):
    store = SharedMemory(tmp_path / "claims.sqlite")
    yield store
    store.close()


def _policy(memory):
    return ClaimPolicy(SimpleNamespace(repo=object()), memory, ttl_minutes=1)


def test_live_claim_can_only_be_renewed_by_its_owner(memory):
    assert memory.claim_issue(17, "agent-a", 60, "agent/a/17")
    before = memory.active_claims()[17].expires_at

    assert memory.renew_claim(17, "agent-a", 120)
    after = memory.active_claims()[17].expires_at
    assert after > before
    assert not memory.renew_claim(17, "agent-b", 120)


def test_expired_claim_cannot_be_revived(memory):
    assert memory.claim_issue(18, "agent-a", -1, "agent/a/18")
    assert not memory.renew_claim(18, "agent-a", 60)
    assert 18 not in memory.active_claims()
    assert 18 in memory.all_active_claims()


def test_expired_claim_is_swept_even_if_agent_heartbeat_is_fresh(memory, monkeypatch):
    released = []
    monkeypatch.setattr(
        claims_module.issue_ops, "release_issue",
        lambda repo, issue, agent, reason: released.append((issue, agent, reason)),
    )
    memory.heartbeat("agent-a", "coder", 1, "working")
    assert memory.claim_issue(19, "agent-a", -1, "agent/a/19")

    freed = _policy(memory).sweep_stale(stale_after_s=3600)

    assert freed == [19]
    assert 19 not in memory.all_active_claims()
    assert released and released[0][0:2] == (19, "agent-a")


def test_live_claim_is_swept_when_owner_heartbeat_is_missing(memory, monkeypatch):
    released = []
    monkeypatch.setattr(
        claims_module.issue_ops, "release_issue",
        lambda repo, issue, agent, reason: released.append(issue),
    )
    assert memory.claim_issue(20, "agent-a", 3600, "agent/a/20")

    assert _policy(memory).sweep_stale(stale_after_s=1) == [20]
    assert released == [20]


def test_agent_claim_renewal_and_shutdown_release(memory, monkeypatch):
    released = []
    monkeypatch.setattr(
        claims_module.issue_ops, "release_issue",
        lambda repo, issue, agent, reason: released.append((issue, agent)),
    )
    policy = _policy(memory)
    assert memory.claim_issue(21, "agent-a", 60, "agent/a/21")
    assert memory.claim_issue(22, "agent-b", 60, "agent/b/22")

    assert policy.renew_agent_claims("agent-a") == []
    assert policy.release_agent_claims("agent-a", "test shutdown") == [21]
    assert 21 not in memory.all_active_claims()
    assert 22 in memory.all_active_claims()
    assert released == [(21, "agent-a")]


def test_reflect_releases_claim_when_cycle_finishes_without_pr():
    released = []
    ctx = SimpleNamespace(
        agent_id="agent-a",
        role=Role.REVIEWER,
        cycle=3,
        claims=SimpleNamespace(
            release=lambda *args: released.append(args),
        ),
        meta={"personal": SimpleNamespace(add_lesson=lambda *args: None)},
        memory=SimpleNamespace(add_decision=lambda *args: None),
        heartbeat=lambda status: None,
    )

    node_reflect(ctx, {"issue_number": 23, "summary": "planning failed"})

    assert released == [(23, "agent-a", "cycle ended without a PR")]


def test_reflect_does_not_release_completed_pr_claim():
    released = []
    ctx = SimpleNamespace(
        agent_id="agent-a",
        role=Role.REVIEWER,
        cycle=4,
        claims=SimpleNamespace(release=lambda *args: released.append(args)),
        meta={"personal": SimpleNamespace(add_lesson=lambda *args: None)},
        memory=SimpleNamespace(add_decision=lambda *args: None),
        heartbeat=lambda status: None,
    )

    node_reflect(ctx, {"issue_number": 24, "pr_number": 8, "summary": "PR opened"})

    assert released == []

def test_complete_claim_marks_done_and_removes_github_label(memory, monkeypatch):
    released = []
    monkeypatch.setattr(
        claims_module.issue_ops, "release_issue",
        lambda repo, issue, agent, reason: released.append((issue, agent, reason)),
    )
    policy = _policy(memory)
    assert memory.claim_issue(25, "agent-a", 60, "agent/a/25")

    policy.complete(25, "agent-a", "agent/a/25")

    row = memory._conn.execute(
        "SELECT status, branch FROM claims WHERE issue=?", (25,)
    ).fetchone()
    assert row == ("done", "agent/a/25")
    assert 25 not in memory.claims_for_agent("agent-a")
    assert released and released[0][0:2] == (25, "agent-a")

def test_partial_github_claim_failure_rolls_back_label_and_local_lease(
    memory, monkeypatch
):
    released = []
    monkeypatch.setattr(
        claims_module.issue_ops, "claim_issue",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("comment API failed")),
    )
    monkeypatch.setattr(
        claims_module.issue_ops, "release_issue",
        lambda repo, issue, agent, reason: released.append((issue, agent, reason)),
    )

    assert not _policy(memory).try_claim(26, "agent-a", "agent/a/26")

    row = memory._conn.execute(
        "SELECT status FROM claims WHERE issue=?", (26,)
    ).fetchone()
    assert row == ("released",)
    assert released and released[0][0:2] == (26, "agent-a")
