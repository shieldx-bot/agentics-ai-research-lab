"""P4 — serial merge queue, held by the Integrator role."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agentics.github_ops.client import GitHubClient
from agentics.memory.shared_memory import SharedMemory
from agentics.github_ops import pr_ops

log = logging.getLogger("policy.mergeq")


@dataclass
class QueueItem:
    pr_number: int
    author: str
    enqueued_at: float


class MergeQueue:
    """FIFO queue persisted in shared memory `decisions` log + in-memory snapshot.

    Only the integrator role may drain the queue; entries are enqueued when a PR
    receives cross-review approval.
    """

    def __init__(self, gh: GitHubClient, memory: SharedMemory, auto_merge: bool = True) -> None:
        self.repo = gh.repo
        self.memory = memory
        self.auto_merge = auto_merge
        self._queue: list[QueueItem] = []

    def enqueue(self, pr_number: int, author: str) -> None:
        if any(it.pr_number == pr_number for it in self._queue):
            return
        self._queue.append(QueueItem(pr_number, author, time.time()))
        self.memory.add_decision("merge-queue", f"enqueue PR #{pr_number} by {author}")

    def pending(self) -> list[QueueItem]:
        return list(self._queue)

    def drain(self, integrator_id: str) -> int:
        """Merge ready PRs one at a time (serial — never parallel merges)."""
        merged = 0
        while self._queue:
            item = self._queue.pop(0)
            ok = pr_ops.merge_when_ready(
                self.repo, item.pr_number, item.author,
                reviewer_agent=integrator_id, auto_merge=self.auto_merge,
            )
            self.memory.add_decision(
                integrator_id,
                f"merge PR #{item.pr_number} by {item.author}",
                "merged" if ok else "deferred (no approval / conflict)",
            )
            if ok:
                merged += 1
        return merged
