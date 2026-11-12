"""Local-only execution layer — the lab runs entirely on one machine.

Replaces every GitHub touchpoint of the research cycle:

    GitHub concept          local replacement
    ---------------------   -----------------------------------------------
    remote repository       a bare git repo at data/lab.git ("origin")
    issues                  the `tasks` table in shared memory
    claim labels            the `claims` table only (single-layer, P1)
    pull requests           the `reviews` table ("MR" — merge request)
    PR reviews / approvals  verdicts written by Reviewer agents to `reviews`
    merge + push to main    a local rebase-style merge pushed to data/lab.git

Nothing in this module performs a network call. The `github_ops` package stays
on disk as dormant legacy code (still covered by its own unit tests) but is no
longer imported by the research cycle.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from agentics.config import PROJECT_ROOT, Settings
from agentics.memory.shared_memory import SharedMemory

log = logging.getLogger("local.ops")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# First line of every hook we manage — hooks without it are owner property.
PRE_RECEIVE_SENTINEL = "# agentics pre-receive (managed)"


def _pre_receive_script(paths: list[str] | tuple[str, ...]) -> str:
    """POSIX-sh pre-receive hook blocking pushes that introduce ``paths``.

    ``git diff --name-only`` output is matched against escaped literal path
    prefixes (guard.is_protected semantics: exact or prefix match), so a
    settings entry like ``configs/`` covers ``configs/secrets.yaml`` too.
    """
    import re as _re

    alts = "|".join(_re.escape(p) for p in paths)
    return f'''#!/bin/sh
{PRE_RECEIVE_SENTINEL} — P10 defence in depth for the local origin (data/lab.git).
# Installed by agentics.local_ops.LocalRepo; regenerated on lab boot when the
# protected-path list changes. Refuses ANY push introducing protected lab
# paths — including direct pushes that bypass workspaces, reviews and the
# merge queue. Owner override: ALLOW_PROTECTED_PATHS=1 in the pushing shell.
ZERO=0000000000000000000000000000000000000000
case "$ALLOW_PROTECTED_PATHS" in
    1|true|yes|on) exit 0 ;;
esac
status=0
while read old new ref; do
    [ "$new" = "$ZERO" ] && continue          # branch deletion
    if [ "$old" = "$ZERO" ]; then
        range="$new --not --all"
    else
        range="$old..$new"
    fi
    bad=$(git diff --name-only $range 2>/dev/null | grep -E '^({alts})')
    if [ -n "$bad" ]; then
        echo "push refused — protected lab paths cannot enter the local origin (P10):" >&2
        echo "$bad" | head -5 >&2
        status=1
    fi
done
exit $status
'''


def branch_name(agent_id: str, task_id: int, title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")[:40]
    return f"agent/{agent_id}/task-{task_id}-{slug}"


# Angles the Supervisor rotates through when the backlog runs dry, so the
# infinite research loop always has somewhere to go.
SURVEY_ANGLES: tuple[str, ...] = (
    "State-of-the-art survey",
    "Benchmark design and metrics",
    "Empirical comparison of methods",
    "Failure-mode analysis",
    "Reproducibility checklist",
    "Tooling and evaluation harness",
)


class LocalRepo:
    """The shared bare repository that plays the role of "origin" locally."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path: Path = settings.local_repo_path

    def ensure(self, src: Path | None = None) -> Path:
        """Seed the bare repo once from the running lab checkout."""
        if (self.path / "HEAD").exists():
            self.install_pre_receive()          # repair path for existing repos
            return self.path
        src = src or PROJECT_ROOT
        if not (src / ".git").exists():
            raise RuntimeError(
                f"cannot seed the local repo: {src} is not a git checkout. "
                "Run the lab from inside its git clone."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.info("seeding local bare repo %s from %s", self.path, src)
        proc = subprocess.run(
            ["git", "clone", "--bare", str(src), str(self.path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"could not seed local repo: {proc.stderr[:400]}")
        # The lab checkout may carry extra refs; keep the bare repo clean and
        # make sure HEAD points at the default branch (main).
        subprocess.run(
            ["git", "-C", str(self.path), "symbolic-ref", "HEAD", "refs/heads/main"],
            capture_output=True, text=True,
        )
        self.install_pre_receive()
        return self.path

    # ------------------------------------------------------------------ P10
    def install_pre_receive(self) -> Path:
        """Defence in depth BELOW the app-level P10 gates (ported from the
        earlier local-mode proposal on this branch).

        The three app gates (workspace write, pre-push diff scan, review +
        merge) only see traffic that flows through agentics code. A direct
        ``git push`` to data/lab.git bypasses all of them — this pre-receive
        hook closes that hole at the git layer: any push introducing a
        protected path is refused before a branch or MR can even exist.

        The pattern is generated from the CURRENT settings (lab.yaml can
        extend the default list); ``ALWAYS_PROTECTED`` credentials/secret
        paths are always included. Only hooks carrying the agentics sentinel
        are (re)written — an owner-customised hook is never clobbered.
        The ``ALLOW_PROTECTED_PATHS=1`` env switch of the PUSHING process is
        honoured (receive-pack inherits it for local pushes), mirroring the
        app-layer override semantics.
        """
        from agentics.policies.guard import ALWAYS_PROTECTED

        base = tuple(getattr(self.settings, "protected_paths", None) or ())
        paths = [p for p in (*base, *ALWAYS_PROTECTED) if p]
        script = _pre_receive_script(paths)
        hook = self.path / "hooks" / "pre-receive"
        hook.parent.mkdir(parents=True, exist_ok=True)
        if hook.exists():
            current = hook.read_text(encoding="utf-8", errors="replace")
            if current == script:
                return hook                     # already up to date
            if PRE_RECEIVE_SENTINEL not in current[:256]:
                log.info("pre-existing owner hook at %s left untouched", hook)
                return hook
        hook.write_text(script, encoding="utf-8")
        try:
            hook.chmod(0o755)
        except OSError:
            pass
        log.info("pre-receive P10 hook installed at %s", hook)
        return hook

    def main_head(self) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.path), "rev-parse", "main"],
            capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()

    def branch_exists(self, branch: str) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.path), "rev-parse", "--verify", branch],
            capture_output=True, text=True, check=False,
        )
        return proc.returncode == 0


class TaskSource:
    """Owner + Supervisor feed tasks here; agents scan and claim them."""

    def __init__(self, memory: SharedMemory, settings: Settings) -> None:
        self.memory = memory
        self.settings = settings

    def ensure_backlog(self) -> None:
        """Seed starter tasks once so a fresh local lab has work immediately."""
        if self.memory.scan_open_tasks(limit=1):
            return
        topic = self.settings.research_topic
        seeds = [
            (f"State-of-the-art survey: {topic}", "research",
             "Survey the current SOTA with citations; store findings in shared "
             "memory so later tasks can build on them."),
            (f"Benchmark design and metrics: {topic}", "code",
             "Design a small, runnable benchmark skeleton under experiments/ "
             "with clear metrics and a README; keep tests green."),
            (f"Reproducibility checklist: {topic}", "docs",
             "Write a reproducibility checklist for experiments under docs/, "
             "referencing the findings stored so far."),
        ]
        for title, category, body in seeds:
            self.memory.create_task(title, body=body, category=category, agent="seed")

    def supervisor_top_up(self) -> int:
        """Supervisor: if nothing is claimable, grow the backlog by one task.

        Two self-generated sources, in priority order (ported from the earlier
        local-mode proposal on this branch — review of cd8b536 found that the
        original top-up always re-opened ``SURVEY_ANGLES[0]``, so a dry backlog
        re-created the same survey forever and never reached the other angles):

          1. failure-driven investigation — the newest failed trace without a
             task yet becomes an investigation task (the GEA loop fed by the
             swarm's own incident history),
          2. the next survey angle — angles ROTATE, least-used first, so the
             infinite research loop revisits the topic from a new side each
             time instead of repeating the first angle.

        Returns the number of tasks created (0 or 1).
        """
        if self.memory.scan_open_tasks(limit=1):
            return 0
        if self._open_failure_investigation():
            return 1
        return self._open_next_survey_angle()

    def _open_failure_investigation(self) -> int:
        """Turn the newest un-investigated failed trace into a bugfix task."""
        known = {t["title"] for t in self.memory.list_tasks(limit=500)}
        for trace in self.memory.recent_traces(limit=5, failures_only=True):
            failure = " ".join(
                str(trace.get("failure") or trace.get("attempted") or "").split()
            )
            if not failure:
                continue
            snippet = failure if len(failure) <= 90 else failure[:89] + "…"
            title = (f"Investigate failure ({trace.get('category', 'unknown')}): "
                     f"{snippet}")
            # a failure already investigated (any state, incl. merged) must not
            # re-open — recurring failures are the evolution task's job
            if title in known:
                continue
            task_id = self.memory.create_task_if_absent(
                title,
                body=(
                    "Self-generated from the group failure traces (GEA).\n\n"
                    f"Agent `{trace.get('agent')}` attempted "
                    f"`{str(trace.get('attempted') or '')[:200]}` -> failed "
                    f"`{failure[:200]}`.\n\n## Task\nInvestigate the root cause, "
                    "fix or document it in the research repo, and record what "
                    "you learned as a trace."
                ),
                category="bugfix",
                agent="supervisor",
            )
            if task_id is not None:
                self.memory.journal("supervisor", "backlog_top_up",
                                    {"task": task_id, "source": "failure_trace",
                                     "trace": trace.get("id")})
                return 1
        return 0

    def _open_next_survey_angle(self) -> int:
        """Open the least-used survey angle for the research topic."""
        topic = self.settings.research_topic
        used: dict[str, int] = {}
        for t in self.memory.list_tasks(limit=500):
            for angle in SURVEY_ANGLES:
                if t["title"].startswith(f"{angle}:"):
                    used[angle] = used.get(angle, 0) + 1
                    break
        angle = min(SURVEY_ANGLES, key=lambda a: (used.get(a, 0), SURVEY_ANGLES.index(a)))
        task_id = self.memory.create_task_if_absent(
            f"{angle}: {topic}",
            body=("Backlog ran dry. Produce the next research increment with "
                  "citations and store findings in shared memory."),
            category="research",
            agent="supervisor",
        )
        if task_id is not None:
            self.memory.journal("supervisor", "backlog_top_up",
                                {"task": task_id, "source": "survey_rotation",
                                 "angle": angle})
            return 1
        return 0


class LocalClaimPolicy:
    """P1 for local mode: the SQLite registry is the single source of truth.

    Mirrors the ClaimPolicy interface (try_claim / release / renew /
    complete / sweep_stale) so graph nodes do not care which mode runs.
    """

    def __init__(self, memory: SharedMemory, ttl_minutes: int) -> None:
        self.memory = memory
        self.ttl_s = ttl_minutes * 60

    def try_claim(self, task_id: int, agent_id: str, branch: str = "") -> bool:
        if not self.memory.claim_issue(task_id, agent_id, self.ttl_s, branch):
            return False
        self.memory.set_task_status(task_id, "claimed", claimed_by=agent_id, branch=branch)
        self.memory.journal(agent_id, "claim", {"task": task_id, "branch": branch})
        return True

    def release(self, task_id: int, agent_id: str, reason: str = "") -> None:
        self.memory.release_claim(task_id, agent_id)
        task = self.memory.get_task(task_id)
        if task and task["status"] in ("claimed", "in_review"):
            self.memory.set_task_status(task_id, "open", claimed_by="")
        self.memory.journal(agent_id, "release", {"task": task_id, "why": reason})

    def renew_agent_claims(self, agent_id: str) -> list[int]:
        lost: list[int] = []
        for task_id, claim in self.memory.claims_for_agent(agent_id).items():
            if not self.memory.renew_claim(task_id, agent_id, self.ttl_s):
                self.release(task_id, agent_id, "lease expired before heartbeat renewal")
                lost.append(task_id)
        return lost

    def release_agent_claims(self, agent_id: str, reason: str = "agent stopped") -> list[int]:
        task_ids = list(self.memory.claims_for_agent(agent_id))
        for task_id in task_ids:
            self.release(task_id, agent_id, reason)
        return task_ids

    def complete(self, task_id: int, agent_id: str, branch: str) -> None:
        """Work handed off to an open MR — the lease ends, the task goes to review."""
        self.memory.complete_claim(task_id, agent_id, branch)
        self.memory.journal(agent_id, "handoff", {"task": task_id, "branch": branch})

    def sweep_stale(self, stale_after_s: float = 900.0) -> list[int]:
        import time as _time
        now = _time.time()
        alive = {a["agent"] for a in self.memory.active_agents(stale_after_s)}
        freed: list[int] = []
        for task_id, claim in self.memory.all_active_claims().items():
            expired = claim.expires_at <= now
            heartbeat_lost = claim.agent not in alive
            if expired or heartbeat_lost:
                reason = ("stale: lease expired" if expired
                          else "stale: agent lost heartbeat")
                self.release(task_id, claim.agent, reason)
                freed.append(task_id)
        return freed


class LocalOps:
    """Facade wiring the local-mode services into one object every agent gets.

    One per Orchestrator (shared): repo, task source, protected-paths guard.
    Per-agent ClaimPolicy / MergeQueue instances are built by the orchestrator.
    """

    def __init__(self, memory: SharedMemory, settings: Settings) -> None:
        from agentics.policies.guard import DEFAULT_PROTECTED_PATHS, ProtectedPathsGuard

        self.memory = memory
        self.settings = settings
        self.repo = LocalRepo(settings)
        self.tasks = TaskSource(memory, settings)
        paths = tuple(getattr(settings, "protected_paths", None) or ())
        self.guard = ProtectedPathsGuard(paths=paths or DEFAULT_PROTECTED_PATHS)

    def setup(self) -> None:
        """One-time preparation: seed the bare repo and the starter backlog."""
        self.repo.ensure()
        self.tasks.ensure_backlog()


class LocalMergeQueue:
    """P4 for local mode — serial merge queue drained by the Integrator.

    `verify(mr_id)` is the tier-2 gate (full tests on main+MR); `merge(mr_id)`
    performs the actual local merge into the bare repo's main. Both are
    supplied by the caller (node_integrate) because they need the Integrator's
    own workspace. When a P10 guard is provided, no MR whose diff touches
    protected paths can ever leave this queue as "merged" — the last of the
    three guard layers.
    """

    MAX_ATTEMPTS = 3

    def __init__(self, memory: SharedMemory, auto_merge: bool = True,
                 guard=None) -> None:
        self.memory = memory
        self.auto_merge = auto_merge
        self.guard = guard

    def enqueue(self, mr_id: int, author: str) -> None:
        if self.memory.mq_enqueue(mr_id, author):
            self.memory.add_decision("merge-queue", f"enqueue MR #{mr_id} by {author}")

    def pending(self) -> list[dict]:
        return self.memory.mq_pending()

    def drain(self, integrator_id: str, verify=None, merge=None) -> int:
        from agentics.memory import lifemem

        merged = 0
        for item in self.pending():
            mr_id = item["pr_number"]
            author = item["author"]
            review = self.memory.get_review(mr_id)
            if review is None:
                self._fail(item, integrator_id, "MR record vanished")
                continue
            if self.guard is not None:
                violations = self.guard.check_task(
                    self.memory, review.get("task_id"), review.get("files", []))
                if violations:
                    self._fail(
                        item, integrator_id,
                        f"P10 protected paths in diff: {violations[:5]} "
                        "— owner-only code cannot be merged")
                    continue
            if verify is not None:
                ok, note = verify(mr_id)
                if not ok:
                    self._fail(item, integrator_id, f"tier-2 verification failed: {note}")
                    continue
            if not self.auto_merge:
                # AUTO_MERGE off: the owner inspects and merges manually — the
                # queue just records readiness and leaves the MR approved.
                self.memory.set_review_verdict(mr_id, "approved", integrator_id)
                self.memory.add_decision(
                    integrator_id, f"MR #{mr_id} by {author} ready (manual merge)",
                    "AUTO_MERGE off")
                continue
            if merge is None:
                # no merge callback wired in this process: leave it queued for
                # an Integrator that has one
                self.memory.mq_mark(mr_id, "queued", bump_attempt=True)
                continue
            if bool(merge(mr_id)):
                self.memory.mq_mark(mr_id, "merged")
                self.memory.set_review_verdict(mr_id, "merged", integrator_id)
                trace_ids = self.memory.set_trace_outcome(mr_id, "merged")
                self.memory.activate_tools(mr_id)
                try:
                    lifemem.on_merge(self.memory, trace_ids)
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not distill skills after merging MR #%s: %s",
                                mr_id, exc)
                review = self.memory.get_review(mr_id)
                if review and review.get("task_id"):
                    self.memory.set_task_status(review["task_id"], "merged")
                self.memory.add_decision(
                    integrator_id, f"merge MR #{mr_id} by {author}", "merged")
                self.memory.journal(integrator_id, "merge",
                                    {"mr": mr_id, "author": author})
                merged += 1
            elif item["attempts"] + 1 >= self.MAX_ATTEMPTS:
                self._fail(item, integrator_id,
                           f"not mergeable after {self.MAX_ATTEMPTS} attempts")
            else:
                self.memory.mq_mark(mr_id, "queued", bump_attempt=True)
                self.memory.add_decision(
                    integrator_id, f"merge MR #{mr_id} by {author}",
                    "deferred (no approval / conflict)")
        return merged

    def _fail(self, item: dict, integrator_id: str, reason: str) -> None:
        mr_id = item["pr_number"]
        self.memory.mq_mark(mr_id, "failed")
        self.memory.set_review_verdict(mr_id, "failed", integrator_id, reason)
        self.memory.set_trace_outcome(mr_id, "failed", reason)
        self.memory.drop_tools(mr_id)
        review = self.memory.get_review(mr_id)
        if review and review.get("task_id"):
            self.memory.set_task_status(review["task_id"], "failed")
        self.memory.add_decision(
            integrator_id, f"reject merge of MR #{mr_id} by {item['author']}",
            reason[:200])
        self.memory.journal(integrator_id, "merge_failed",
                            {"mr": mr_id, "why": reason[:300]})
