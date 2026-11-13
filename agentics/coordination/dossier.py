"""Reviewer handoff dossier — the verification tier sees the task's history.

Gap closed: the reviewer's prompt carried the diff, the author's recorded
verification and the P16 evidence rows — but NOT the task's own past. An MR
could re-implement an approach that already failed twice on this very task
and the reviewer had no way to know; the failures sat in ``traces`` and the
objections in dead MR rows, consumed by nobody at review time. This is the
inter-agent-misalignment failure of MAST (arXiv:2503.13657) and the missing
"communication bridge" of Cognition's multi-agent writeup: the verifier must
receive the FULL trace of what happened before it, not just the artifact.

The dossier is assembled mechanically (no LLM, no extra cost) from the
shared memory:

* earlier attempts on this task — every trace row keyed by the task id,
  with its outcome and the recorded failure/approach reason;
* prior MR verdicts on this task — ``changes_requested`` / ``rejected`` /
  ``failed`` MRs older than the one being reviewed, with their verdict
  comments (the objections the panel already raised once).

Bounded by construction: a prompt must not drown in history. Empty string
when the task has no history — cold-start prompts stay byte-identical.
"""
from __future__ import annotations

from agentics.memory.shared_memory import SharedMemory

# MR states that carry an objection worth handing to the next reviewer.
OBJECTION_STATES = frozenset({"changes_requested", "rejected", "failed"})


def review_dossier(
    memory: SharedMemory,
    task_id: int | None,
    mr_id: int | None = None,
    max_attempts: int = 3,
    max_objections: int = 3,
    max_chars: int = 1200,
) -> str:
    """Bounded history of ``task_id`` for the reviewer's prompt.

    ``mr_id`` excludes the MR under review itself (its objections are being
    formed right now — only PRIOR verdicts belong in the dossier). Returns
    "" when the task is new: nothing to hand over, nothing to inject.
    """
    if task_id is None:
        return ""
    lines: list[str] = []

    attempts = memory.recent_traces(limit=20, issue=task_id)[-max_attempts:]
    if attempts:
        lines.append("Earlier attempts on this task:")
        for t in attempts:
            reason = t.get("failure") or t.get("attempted") or ""
            lines.append(f"- {t['agent']}: {t['outcome']} — {str(reason)[:140]}")

    objections: list[str] = []
    for mr in memory.reviews_by_task(task_id):
        if mr_id is not None and mr["id"] >= mr_id:
            break                     # only MRs OLDER than the one reviewed
        if mr.get("status") in OBJECTION_STATES:
            objections.append(
                f"!#{mr['id']} {mr['status']}: "
                f"{str(mr.get('verdict_comment') or '')[:140]}")
    if objections:
        lines.append("Prior objections from MR reviews on this task:")
        lines.extend(f"- {o}" for o in objections[-max_objections:])

    if not lines:
        return ""
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
