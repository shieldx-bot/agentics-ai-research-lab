"""AgentContext — everything one agent thread owns (isolation per thread)."""
from __future__ import annotations

from dataclasses import dataclass, field

from agentics.config import Settings
from agentics.github_ops.client import GitHubClient
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.llm.session_host import BrowserSessionHost
from agentics.memory.personal_memory import PersonalMemory
from agentics.memory.shared_memory import SharedMemory
from agentics.memory.vector_store import VectorStore
from agentics.policies.claims import ClaimPolicy
from agentics.policies.merge_queue import MergeQueue
from agentics.policies.ownership import OwnershipPolicy
from agentics.agents.roles import Role


@dataclass
class AgentContext:
    agent_id: str
    role: Role
    settings: Settings
    gh: GitHubClient
    memory: SharedMemory
    vector: VectorStore
    llm: BrowserSessionHost | None
    workspace: RepoWorkspace
    claims: ClaimPolicy
    ownership: OwnershipPolicy
    merge_queue: MergeQueue
    cycle: int = 0
    meta: dict = field(default_factory=dict)

    def heartbeat(self, status: str = "working") -> None:
        self.memory.heartbeat(self.agent_id, self.role.value, self.cycle, status)
