"""Roles & prompt templates. Agents self-select roles from config role_map,
falling back to automatic round-robin assignment at startup."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    REVIEWER = "reviewer"
    INTEGRATOR = "integrator"


ALL_ROLES = list(Role)

ROLE_MISSIONS: dict[Role, str] = {
    Role.SUPERVISOR: (
        "Scan the repo's open issues, judge priorities, create new research "
        "issues when the backlog runs dry, and arbitrate claim disputes. "
        "You do NOT write code yourself."
    ),
    Role.RESEARCHER: (
        "Survey academic literature on the issue topic using academic search "
        "mode, extract findings with real citations (BibTeX), deduplicate "
        "against the swarm's vector memory, and store findings in shared memory."
    ),
    Role.CODER: (
        "Turn research findings into working code/experiments in your own "
        "branch. Write code, run safe commands in the workspace, commit and "
        "push. Respect the ownership matrix."
    ),
    Role.WRITER: (
        "Turn research findings and code results into clean documentation and "
        "research reports (Markdown), with standard citations."
    ),
    Role.REVIEWER: (
        "Cross-review open PRs authored by other agents. Check correctness, "
        "issue alignment, ownership compliance, citation quality, and that no "
        "secrets are committed."
    ),
    Role.INTEGRATOR: (
        "Drain the merge queue serially: verify cross-review approval, update "
        "branch, rebase-merge into main. Request rebase when conflicts appear."
    ),
}

TOPIC_LINE = (
    "Research domain: {topic}\n"
    "You are agent [{agent_id}] playing the role of {role}.\n"
    "Your mission: {mission}\n"
)

MEMORY_CONTEXT_LINE = (
    "Shared memory tail (latest messages from other agents):\n{messages}\n"
    "Related findings already in memory (do not duplicate):\n{findings}\n"
)


def build_system_prompt(agent_id: str, role: Role, topic: str) -> str:
    return (
        TOPIC_LINE.format(topic=topic, agent_id=agent_id, role=role.value,
                          mission=ROLE_MISSIONS[role])
        + "Collaboration rules: every decision you make must be written to "
          "shared memory; never touch another agent's branch; yield when your "
          "diff enters foreign territory; never merge your own PR."
    )


def parse_json_block(text: str) -> dict:
    """Extract the first JSON object from an LLM answer (tolerant)."""
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.removeprefix("json").strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class RoleAssignment:
    agent_id: str
    role: Role


def assign_roles(n_agents: int, role_map: dict[int, str] | None) -> list[Role]:
    """role_map keys are 1-based account indices; missing entries round-robin."""
    role_map = role_map or {}
    roles: list[Role] = []
    for i in range(1, n_agents + 1):
        raw = role_map.get(i, "")
        try:
            roles.append(Role(str(raw).lower()))
        except ValueError:
            roles.append(ALL_ROLES[(i - 1) % len(ALL_ROLES)])
    # guarantee at least one supervisor and one integrator for the machinery
    if Role.SUPERVISOR not in roles and n_agents > 0:
        roles[0] = Role.SUPERVISOR
    if Role.INTEGRATOR not in roles and n_agents > 1:
        roles[-1] = Role.INTEGRATOR
    return roles
