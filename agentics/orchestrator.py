"""Orchestrator — spawns one asyncio task per cookie account.

Each task = one agent thread = one LLM identity, running its LangGraph
research cycle forever (or until the owner's STOP file appears).
"""
from __future__ import annotations

import asyncio
import logging
import random

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, assign_roles
from agentics.config import Settings, load_yaml_overrides, PROJECT_ROOT
from agentics.github_ops.client import GitHubClient
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.llm.pool import AccountPool
from agentics.memory.personal_memory import PersonalMemory
from agentics.memory.shared_memory import SharedMemory
from agentics.memory.vector_store import VectorStore
from agentics.policies.claims import ClaimPolicy
from agentics.policies.merge_queue import MergeQueue
from agentics.policies.ownership import OwnershipPolicy
from agentics.graph.research_graph import run_cycle
from agentics.stop import should_stop

log = logging.getLogger("orchestrator")


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


class Orchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_yaml_overrides(Settings())
        self.settings.ensure_dirs()
        self.memory = SharedMemory(self.settings.db_path)
        self.vector = VectorStore(self.settings.chroma_path)
        self.pool = AccountPool(self.settings)
        self.gh = GitHubClient(self.settings)
        self.ownership = OwnershipPolicy.from_config(None)
        self.agents: list[AgentContext] = []

    # ------------------------------------------------------------ assembly
    def build_contexts(self) -> list[AgentContext]:
        n = min(len(self.pool), self.settings.max_agents) or 0
        if n == 0:
            raise RuntimeError(
                "No cookie accounts found. Put account-01.json… into accounts/ "
                "(see accounts/README.md)."
            )
        roles = assign_roles(n, _role_map(self.settings))
        contexts: list[AgentContext] = []
        for i in range(n):
            agent_id = f"a{i + 1:02d}"
            account = self.pool.accounts[i]
            personal = PersonalMemory(agent_id, self.settings.data_dir / "personal")
            ctx = AgentContext(
                agent_id=agent_id,
                role=roles[i],
                settings=self.settings,
                gh=self.gh,
                memory=self.memory,
                vector=self.vector,
                llm=self.pool.client_for(account),
                workspace=RepoWorkspace(agent_id, self.settings),
                claims=ClaimPolicy(self.gh, self.memory, self.settings.claim_ttl_minutes),
                ownership=self.ownership,
                merge_queue=MergeQueue(self.gh, self.memory, self.settings.auto_merge),
                meta={"account": account.name, "personal": personal},
            )
            contexts.append(ctx)
        self.agents = contexts
        # announce roles once so agents "know each other"
        for ctx in contexts:
            self.memory.post_message(
                "orchestrator", "broadcast",
                f"{ctx.agent_id} started as {ctx.role.value} "
                f"(account: {ctx.meta['account']})"
            )
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

    async def _agent_loop(self, ctx: AgentContext) -> None:
        tag = f"{ctx.agent_id}/{ctx.role.value}"
        log.info("[%s] thread up", tag)
        heartbeat_task = asyncio.create_task(self._lease_heartbeat(ctx))
        try:
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
                    # cookie trouble → cooldown & maybe swap account
                    if ctx.llm is not None and ("4" in str(exc)[:2] or "auth" in str(exc).lower()):
                        self.pool.mark_failed(ctx.llm.account, str(exc))
                        nxt = self.pool.lease()
                        if nxt is not None:
                            ctx.llm = self.pool.client_for(nxt)
                            log.info("[%s] switched to account %s", tag, nxt.name)
                pause = self.settings.cycle_pause_seconds + random.uniform(0, 15)
                for _ in range(int(pause)):
                    if should_stop():
                        break
                    await asyncio.sleep(1)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                await asyncio.to_thread(
                    ctx.claims.release_agent_claims, ctx.agent_id, "agent shutdown"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] shutdown claim cleanup failed: %s", tag, exc)
            ctx.heartbeat("stopped")
            ctx.meta["personal"].close()
            log.info("[%s] thread stopped gracefully", tag)

    async def run_forever(self) -> None:
        contexts = self.build_contexts()
        log.info("starting %d agent threads (roles: %s)",
                 len(contexts), [c.role.value for c in contexts])
        tasks = [asyncio.create_task(self._agent_loop(c)) for c in contexts]
        try:
            await asyncio.gather(*tasks)
        finally:
            self.memory.close()

    def status(self) -> str:
        lines = ["== agents =="]
        for a in self.memory.active_agents(stale_after_s=86400):
            lines.append(
                f"{a['agent']:<6} role={a['role']:<12} cycle={a['cycle']:<4} "
                f"status={a['status']}"
            )
        lines.append("== accounts ==")
        lines.extend(self.pool.status_lines())
        lines.append("== claims ==")
        for issue_no, c in self.memory.active_claims().items():
            lines.append(f"issue #{issue_no} -> {c.agent} (branch {c.branch})")
        return "\n".join(lines) or "(no data yet)"
