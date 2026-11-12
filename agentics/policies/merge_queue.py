"""P4 — serial merge queue, held by the Integrator role.

The queue lives in shared memory (table ``merge_queue``), NOT in the
`MergeQueue` instance: every agent owns its own instance, so an in-RAM list
would mean the Reviewer enqueues into one queue and the Integrator drains a
different, empty one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from agentics.github_ops import pr_ops
from agentics.github_ops.client import GitHubClient
from agentics.memory import lifemem
from agentics.memory.shared_memory import SharedMemory

log = logging.getLogger("policy.mergeq")

# (pr_number) -> (ok, note). Tier-2 verification, run on main+PR just before merge.
Verifier = Callable[[int], tuple[bool, str]]


@dataclass
class QueueItem:
    pr_number: int
    author: str
    enqueued_at: float
    attempts: int = 0


class MergeQueue:
    """FIFO queue persisted in shared memory; only the integrator drains it.

    Entries are enqueued when a PR receives cross-review approval.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, gh: GitHubClient, memory: SharedMemory, auto_merge: bool = True) -> None:
        self.repo = gh.repo
        self.memory = memory
        self.auto_merge = auto_merge

    def enqueue(self, pr_number: int, author: str) -> None:
        if self.memory.mq_enqueue(pr_number, author):
            self.memory.add_decision("merge-queue", f"enqueue PR #{pr_number} by {author}")

    def pending(self) -> list[QueueItem]:
        return [QueueItem(r["pr_number"], r["author"], r["enqueued_at"], r["attempts"])
                for r in self.memory.mq_pending()]

    def drain(self, integrator_id: str, verify: Verifier | None = None) -> int:
        """Merge ready PRs one at a time (serial — never parallel merges).

        `verify` is the tier-2 gate: it must pass on (main + PR) right before the
        merge, which catches "green alone, red after rebase".
        """
        merged = 0
        for item in self.pending():
            if verify is not None:
                ok, note = verify(item.pr_number)
                if not ok:
                    self._fail(item, integrator_id, f"tier-2 verification failed: {note}")
                    continue
            ok = pr_ops.merge_when_ready(
                self.repo, item.pr_number, item.author,
                reviewer_agent=integrator_id, auto_merge=self.auto_merge,
            )
            if ok:
                self.memory.mq_mark(item.pr_number, "merged")
                trace_ids = self.memory.set_trace_outcome(item.pr_number, "merged")
                self.memory.activate_tools(item.pr_number)
                try:      # LifeMem: merged work becomes a distilled skill for everyone
                    lifemem.on_merge(self.memory, trace_ids)
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not distill skills after merging PR #%s: %s",
                                item.pr_number, exc)
                self.memory.add_decision(
                    integrator_id, f"merge PR #{item.pr_number} by {item.author}", "merged")
                merged += 1
            elif item.attempts + 1 >= self.MAX_ATTEMPTS:
                self._fail(item, integrator_id,
                           f"not mergeable after {self.MAX_ATTEMPTS} attempts (conflict/approval)")
            else:
                self.memory.mq_mark(item.pr_number, "queued", bump_attempt=True)
                self.memory.add_decision(
                    integrator_id, f"merge PR #{item.pr_number} by {item.author}",
                    "deferred (no approval / conflict)")
        return merged

    def _fail(self, item: QueueItem, integrator_id: str, reason: str) -> None:
        self.memory.mq_mark(item.pr_number, "failed")
        self.memory.set_trace_outcome(item.pr_number, "failed", reason)
        self.memory.drop_tools(item.pr_number)
        self.memory.add_decision(
            integrator_id, f"reject merge of PR #{item.pr_number} by {item.author}", reason[:200])
        try:
            self.repo.get_pull(item.pr_number).create_issue_comment(
                f"Integrator ({integrator_id}): {reason}")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not comment on PR #%s: %s", item.pr_number, exc)
