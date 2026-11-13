"""Orchestrator — spawns one asyncio task per cookie account.

Each task = one agent thread = one LLM identity, running its LangGraph
research cycle forever (or until the owner's STOP file appears).

Local-only mode: everything runs on this machine — tasks/reviews in SQLite,
git in a shared bare repo at data/lab.git, zero GitHub calls.
"""
from __future__ import annotations

import asyncio
import logging
import random

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, assign_roles
from agentics.config import Settings, load_yaml_overrides, PROJECT_ROOT
# NOTE: github_ops stays dormant in local mode, but the process still
# configures its shared scan cache so a re-enabled GitHub mode inherits the
# production pacing settings (port of 408e7a5).
from agentics.github_ops import issue_ops
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.llm.errors import is_auth_failure
from agentics.llm.session_host import BrowserSessionHost
from agentics.local_ops import LocalClaimPolicy, LocalMergeQueue, LocalOps
from agentics.memory.personal_memory import PersonalMemory
from agentics.memory.shared_memory import SharedMemory
from agentics.memory.vector_store import VectorStore
from agentics.policies.ownership import OwnershipPolicy
from agentics.graph.research_graph import run_cycle
from agentics.stop import should_stop

log = logging.getLogger("orchestrator")


def _effective_agent_count(settings: Settings, available_accounts: int) -> int:
    """Respect the Hyperbrowser client's single-session architecture."""
    count = min(max(0, available_accounts), max(0, settings.max_agents))
    if (settings.browser_backend or "local").lower() == "hyperbrowser" and count > 1:
        log.warning(
            "BROWSER_BACKEND=hyperbrowser currently supports one active session; "
            "limiting this process to 1 agent (requested %d).",
            count,
        )
        return 1
    return count


def _role_map(settings: Settings) -> dict[int, str]:
    cfg_path = PROJECT_ROOT / "configs" / "lab.yaml"
    role_map: dict[int, str] = {}
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw = data.get("role_map") or {}
        role_map = {int(k): str(v) for k, v in raw.items()}
    return role_map


def _startup_delay_s(index: int) -> float:
    """Deterministic per-agent startup stagger.

    Ten agents booting at once fire ten synchronized issue scans + claim
    attempts — exactly the burst that trips GitHub secondary rate limits
    (403 storms). Spreading first cycles over ~1 minute keeps volume flat.
    """
    return float(min(max(0, index) * 7, 60))


class Orchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_yaml_overrides(Settings())
        self.settings.ensure_dirs()
        # Apply the YAML/env scan-cache TTL to the shared issue cache.
        issue_ops.set_scan_cache_ttl(getattr(self.settings, "gh_scan_cache_ttl", 60))
        self.memory = SharedMemory(self.settings.db_path)
        self.vector = VectorStore(self.settings.chroma_path)
        self.host = BrowserSessionHost.from_settings(self.settings)
        self.local = LocalOps(self.memory, self.settings)
        self.local.setup()          # seed data/lab.git + the starter backlog
        self.ownership = OwnershipPolicy.from_config(None)
        self.agents: list[AgentContext] = []
        self._integrity_baseline()

    # ------------------------------------------------------------ integrity
    def _integrity_baseline(self) -> None:
        """DETECT layer: hash the tree the lab itself runs from at startup.

        P10 changeset protection (guard.py) stops AGENT writes into the
        engine; this catches everything else — a botched merge, an escaped
        process, a manual edit. Drift is an owner-visible alert, never a
        cycle-breaking error.
        """
        try:
            from agentics.sandbox.integrity import IntegrityGuard

            guard = IntegrityGuard(PROJECT_ROOT, self.settings.integrity_path)
            existed, drift = guard.ensure_baseline()
            if drift:
                msg = (
                    "Runtime integrity drift: %d file(s) differ from the baseline "
                    "- run `agentics verify` (owner decision)" % len(drift)
                )
                self.memory.journal("orchestrator", "integrity_alert",
                                    {"drift": drift[:10]})
                self.memory.post_message("orchestrator", "alert", msg)
                log.warning("runtime integrity drift: %d file(s) - see `agentics verify`",
                            len(drift))
            elif not existed:
                log.info("integrity baseline created (first run)")
        except Exception as exc:  # noqa: BLE001
            log.warning("integrity baseline failed: %s", exc)

    # ------------------------------------------------------------ assembly
    def build_contexts(self) -> list[AgentContext]:
        n = _effective_agent_count(self.settings, len(self.host))
        if n == 0:
            raise RuntimeError(
                "No browser-session accounts found. Put account-01.json… "
                "(Perplexity) and/or qwen-01.json… (Qwen) into accounts/ "
                "(see accounts/README.md and docs/LLM_PROVIDERS.md)."
            )
        roles = assign_roles(n, _role_map(self.settings))
        contexts: list[AgentContext] = []
        for i in range(n):
            agent_id = f"a{i + 1:02d}"
            # agent i is pinned to account i of each pool (1 thread = 1 identity)
            pinned = self.host.view(prefer_index=i)
            pplx_accs = self.host.pools["perplexity"].accounts
            qwen_accs = self.host.pools["qwen"].accounts
            first = (pplx_accs[i] if i < len(pplx_accs)
                     else qwen_accs[i] if i < len(qwen_accs) else None)
            personal = PersonalMemory(agent_id, self.settings.data_dir / "personal")
            ctx = AgentContext(
                agent_id=agent_id,
                role=roles[i],
                settings=self.settings,
                memory=self.memory,
                vector=self.vector,
                llm=pinned,
                workspace=RepoWorkspace(agent_id, self.settings, guard=self.local.guard),
                ownership=self.ownership,
                local=self.local,
                claims=LocalClaimPolicy(self.memory, self.settings.claim_ttl_minutes),
                merge_queue=LocalMergeQueue(
                    self.memory, self.settings.auto_merge, guard=self.local.guard),
                meta={"account": getattr(first, "name", "shared"), "personal": personal},
            )
            contexts.append(ctx)
        self.agents = contexts
        # announce roles once so agents "know each other" — and so the owner
        # can see who is who from second zero (dashboard / journal)
        for ctx in contexts:
            self.memory.post_message(
                "orchestrator", "broadcast",
                f"{ctx.agent_id} started as {ctx.role.value} "
                f"(account: {ctx.meta['account']})"
            )
            self.memory.journal(ctx.agent_id, "startup", {
                "role": ctx.role.value, "account": ctx.meta["account"],
            })
        return contexts

    # ------------------------------------------------------------ runtime
    async def _lease_heartbeat(self, ctx: AgentContext) -> None:
        """Keep the agent liveness record and its active issue leases fresh."""
        interval = max(1.0, min(ctx.claims.ttl_s / 3.0, 30.0))
        while True:
            await asyncio.sleep(interval)
            try:
                ctx.heartbeat("working")
                lost = await asyncio.to_thread(
                    ctx.claims.renew_agent_claims, ctx.agent_id
                )
                if lost:
                    ctx.memory.post_message(
                        ctx.agent_id, "alert",
                        f"Lost expired claim lease(s): {lost}",
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] lease heartbeat failed: %s", ctx.agent_id, exc)

    async def _agent_loop(self, ctx: AgentContext, index: int = 0) -> None:
        tag = f"{ctx.agent_id}/{ctx.role.value}"
        log.info("[%s] thread up", tag)
        heartbeat_task = asyncio.create_task(self._lease_heartbeat(ctx))
        try:
            delay = _startup_delay_s(index)
            if delay > 0 and not should_stop():
                log.info("[%s] staggered start in %.0fs (anti-burst)", tag, delay)
                waited = 0.0
                while waited < delay and not should_stop():
                    await asyncio.sleep(min(1.0, delay - waited))
                    waited += 1.0
            while not should_stop():
                try:
                    ctx.heartbeat("cycle-start")
                    await run_cycle(ctx)
                    ctx.heartbeat("cycle-end")
                except Exception as exc:  # noqa: BLE001
                    log.exception("[%s] cycle failed: %s", tag, exc)
                    ctx.memory.post_message(ctx.agent_id, "alert", f"cycle error: {exc}")
                    # A failed graph invocation cannot safely retain its issue lease.
                    try:
                        await asyncio.to_thread(
                            ctx.claims.release_agent_claims,
                            ctx.agent_id,
                            f"cycle failed: {type(exc).__name__}",
                        )
                    except Exception as release_exc:  # noqa: BLE001
                        log.warning("[%s] claim cleanup failed: %s", tag, release_exc)
                    # cookie trouble -> cooldown the account that served the
                    # call; the host re-leases a healthy one automatically at
                    # the next ask. is_auth_failure classifies precisely so a
                    # generic cycle error that merely starts with "4" (or an
                    # unrelated "auth" substring) cannot cool a healthy cookie.
                    # Graph-node LLM failures degrade to state["error"] before
                    # reaching this branch (ported from main); only errors that
                    # still bubble up (heartbeat, non-graph code) land here, and
                    # only genuine auth failures cool a cookie.
                    if ctx.llm is not None and is_auth_failure(exc):
                        ctx.llm.mark_failed(str(exc))
                pause = self.settings.cycle_pause_seconds + random.uniform(0, 15)
                for _ in range(int(pause)):
                    if should_stop():
                        break
                    await asyncio.sleep(1)
        finally:
            try:
                heartbeat_task.cancel()
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] heartbeat cleanup failed: %s", tag, exc)
            # Critical cleanup must survive the task's own cancellation (it
            # runs during Ctrl-C shutdown, when the task is being cancelled).
            try:
                await asyncio.shield(
                    asyncio.to_thread(
                        ctx.claims.release_agent_claims, ctx.agent_id, "agent shutdown"
                    )
                )
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                log.warning("[%s] shutdown claim cleanup failed: %s", tag, exc)
            try:
                ctx.heartbeat("stopped")
                ctx.meta["personal"].close()
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] shutdown bookkeeping failed: %s", tag, exc)
            log.info("[%s] thread stopped gracefully", tag)

    async def _integrity_loop(self) -> None:
        """P10 layer 4 — self-healing: re-hash the protected trees inside every
        agent workspace; drift (a rogue script that bypassed the action-time
        gates) is restored from git and journalled for the dashboard."""
        from agentics.policies.integrity import IntegrityWatcher

        watcher = IntegrityWatcher(
            self.memory, self.local.guard, self.settings.workspaces_dir)
        await watcher.run(self.settings.integrity_interval_s)

    async def run_forever(self) -> None:
        contexts = self.build_contexts()
        log.info("starting %d agent threads (roles: %s)",
                 len(contexts), [c.role.value for c in contexts])
        tasks = [
            asyncio.create_task(self._agent_loop(c, i))
            for i, c in enumerate(contexts)
        ]
        watcher_task = None
        if int(getattr(self.settings, "integrity_interval_s", 0) or 0) > 0:
            watcher_task = asyncio.create_task(self._integrity_loop())
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass                    # Ctrl-C -> asyncio.run cancels this task
        finally:
            # Deterministic teardown: cancel every agent thread and WAIT for
            # the cancellations to land before closing resources. Without the
            # awaited gather, asyncio destroys still-pending tasks and dumps
            # the "Task was destroyed but it is pending!" noise seen at shutdown.
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if watcher_task is not None:
                watcher_task.cancel()
                await asyncio.gather(watcher_task, return_exceptions=True)
            log.info("all agent threads stopped, closing sessions")
            try:
                await self.host.close_all()   # stop cloud sessions / sockets
            except Exception as exc:  # noqa: BLE001
                log.warning("session close during shutdown failed: %s", exc)
            self.memory.close()

    def status(self) -> str:
        lines = ["== agents =="]
        for a in self.memory.active_agents(stale_after_s=86400):
            task = f" task #{a['task_id']}: {a['task_title'][:40]}" if a.get("task_id") else ""
            lines.append(
                f"{a['agent']:<6} role={a['role']:<12} cycle={a['cycle']:<4} "
                f"status={a['status']:<12} node={a['node'] or '-':<10}{task}"
            )
        lines.append("== tasks ==")
        counts = self.memory.task_counts()
        lines.append("counts: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
        lines.append("== claims ==")
        for task_id, c in self.memory.active_claims().items():
            lines.append(f"task #{task_id} -> {c.agent} (branch {c.branch})")
        lines.append("== accounts ==")
        lines.extend(self.host.status_lines())
        return "\n".join(lines) or "(no data yet)"
