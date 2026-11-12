"""P1+P6 — claim coordination on top of shared memory + GitHub labels.

Two-layer lock: shared memory (fast, local, atomic) + GitHub label (visible to
everyone). An issue may only be worked on when BOTH agree the claim is free.
"""
from __future__ import annotations

import logging
import time

from agentics.github_ops import issue_ops
from agentics.github_ops.client import GitHubClient
from agentics.memory.shared_memory import SharedMemory

log = logging.getLogger("policy.claims")


class ClaimPolicy:
    def __init__(self, gh: GitHubClient, memory: SharedMemory, ttl_minutes: int) -> None:
        self.repo = gh.repo
        self.memory = memory
        self.ttl_s = ttl_minutes * 60

    def try_claim(self, issue_number: int, agent_id: str, branch: str = "") -> bool:
        # Layer 1: local registry (atomic in SQLite)
        if not self.memory.claim_issue(issue_number, agent_id, self.ttl_s, branch):
            return False
        # Layer 2: GitHub side (labels + comment)
        try:
            ok = issue_ops.claim_issue(
                self.repo, issue_number, agent_id, ttl_minutes=self.ttl_s // 60
            )
        except Exception as exc:  # noqa: BLE001
            log.error("github claim failed on #%s: %s", issue_number, exc)
            self.memory.release_claim(issue_number, agent_id)
            # The GitHub label may have been added before a comment/API failure.
            try:
                issue_ops.release_issue(
                    self.repo, issue_number, agent_id, "claim acquisition rolled back"
                )
            except Exception as cleanup_exc:  # noqa: BLE001
                log.warning("claim rollback cleanup failed on #%s: %s",
                            issue_number, cleanup_exc)
            return False
        if not ok:
            self.memory.release_claim(issue_number, agent_id)
            return False
        return True

    def release(self, issue_number: int, agent_id: str, reason: str = "") -> None:
        self.memory.release_claim(issue_number, agent_id)
        try:
            issue_ops.release_issue(self.repo, issue_number, agent_id, reason)
        except Exception as exc:  # noqa: BLE001
            log.warning("github release failed on #%s: %s", issue_number, exc)

    def renew_agent_claims(self, agent_id: str) -> list[int]:
        """Renew all live leases owned by an agent; return leases already lost."""
        lost: list[int] = []
        for issue_no, claim in self.memory.claims_for_agent(agent_id).items():
            if not self.memory.renew_claim(issue_no, agent_id, self.ttl_s):
                self.release(issue_no, agent_id, "lease expired before heartbeat renewal")
                lost.append(issue_no)
        return lost

    def release_agent_claims(self, agent_id: str, reason: str = "agent stopped") -> list[int]:
        """Release all active claims for an agent, including expired local leases."""
        issue_numbers = list(self.memory.claims_for_agent(agent_id))
        for issue_no in issue_numbers:
            self.release(issue_no, agent_id, reason)
        return issue_numbers

    def complete(self, issue_number: int, agent_id: str, branch: str) -> None:
        """Mark local work complete and remove the visible GitHub claim label."""
        self.memory.complete_claim(issue_number, agent_id, branch)
        try:
            issue_ops.release_issue(
                self.repo, issue_number, agent_id, f"handed off to PR branch {branch}"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("GitHub claim label cleanup failed on #%s: %s", issue_number, exc)

    def sweep_stale(self, stale_after_s: float = 900.0) -> list[int]:
        """Release expired leases or claims whose owner heartbeat has gone stale."""
        now = time.time()
        alive = {a["agent"] for a in self.memory.active_agents(stale_after_s)}
        freed: list[int] = []
        for issue_no, claim in self.memory.all_active_claims().items():
            expired = claim.expires_at <= now
            heartbeat_lost = claim.agent not in alive
            if expired or heartbeat_lost:
                reason = "stale: lease expired" if expired else "stale: agent lost heartbeat"
                self.release(issue_no, claim.agent, reason)
                freed.append(issue_no)
        return freed
