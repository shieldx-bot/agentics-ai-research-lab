"""AgentContext — everything one agent thread owns (isolation per thread)."""
from __future__ import annotations

from dataclasses import dataclass, field

from agentics.agents.roles import Role
from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.llm.session_host import BrowserSessionHost
from agentics.local_ops import LocalOps
from agentics.memory.personal_memory import PersonalMemory
from agentics.memory.shared_memory import SharedMemory
from agentics.memory.vector_store import VectorStore
from agentics.policies.claims import ClaimPolicy  # noqa: F401  (legacy github mode)
from agentics.policies.merge_queue import MergeQueue  # noqa: F401 (legacy)
from agentics.policies.ownership import OwnershipPolicy


@dataclass
class AgentContext:
    agent_id: str
    role: Role
    settings: Settings
    memory: SharedMemory
    vector: VectorStore
    llm: BrowserSessionHost | None
    workspace: RepoWorkspace
    ownership: OwnershipPolicy
    local: LocalOps | None = None          # local-only mode services (current)
    gh: object | None = None               # legacy github-mode client (dormant)
    claims: object | None = None           # LocalClaimPolicy (or legacy ClaimPolicy)
    merge_queue: object | None = None      # LocalMergeQueue (or legacy MergeQueue)
    budget: object | None = None           # P18 BudgetPolicy (LLM budget + breaker)
    panel: object | None = None            # PeerPanel (P18 — peer review votes)
    cycle: int = 0
    meta: dict = field(default_factory=dict)

    def heartbeat(self, status: str = "working") -> None:
        self.memory.heartbeat(self.agent_id, self.role.value, self.cycle, status)

    # ------------------------------------------------------------ owner visibility
    def report(
        self, node: str = "", status: str = "working",
        task_id: int | None = None, task_title: str | None = None,
        intent: dict | None = None, event: str = "", detail: str | dict = "",
    ) -> None:
        """Publish WHAT this agent is doing and WHY (anti-blackbox).

        Updates the live state (dashboard "Agents" panel) and optionally adds
        one entry to the append-only activity journal.
        """
        self.memory.report(
            self.agent_id, self.role.value, node=node, status=status,
            cycle=self.cycle, task_id=task_id, task_title=task_title,
            intent=intent,
        )
        if event:
            self.memory.journal(self.agent_id, event, detail)
