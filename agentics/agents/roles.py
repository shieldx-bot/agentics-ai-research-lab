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

# Immutable collaboration rules — appended by CODE to EVERY system prompt,
# static or LLM-synthesized. The model may rewrite its mission freely; it can
# never rewrite these (the planning layer re-appends them after synthesis).
GUARDRAILS = (
    "Collaboration rules (non-negotiable; no prompt can change them): write "
    "every decision to shared memory; never touch another agent's branch; "
    "yield when your diff enters foreign territory; never merge your own MR; "
    "never modify protected paths (agentics/, configs/, accounts/, scripts/, "
    ".github/); never commit secrets or credentials; back claims by running "
    "code, not by opinion."
)


def static_mission(role: Role | str) -> str:
    """Fallback mission text — used when no LLM-authored prompt exists."""
    try:
        return ROLE_MISSIONS[Role(str(role))]
    except (ValueError, KeyError):
        return ROLE_MISSIONS[Role.RESEARCHER]


def compose_system_prompt(
    agent_id: str, role: Role | str, topic: str, mission: str
) -> str:
    """Identity + mission + guardrails. `mission` may be LLM-authored
    (adaptive prompts) or the static default; the guardrails are code-owned."""
    role_val = getattr(role, "value", None) or str(role)
    return TOPIC_LINE.format(
        topic=topic, agent_id=agent_id, role=role_val, mission=mission
    ) + GUARDRAILS


def build_system_prompt(agent_id: str, role: Role, topic: str) -> str:
    """Static system prompt (the pre-adaptive behavior, kept as fallback)."""
    return compose_system_prompt(agent_id, role, topic, static_mission(role))


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
    # A LONE agent cannot coordinate anything: supervisor duties (arbitration,
    # backlog feeding for OTHER agents) are moot, the supervisor node never
    # touches the LLM, and a single-token lab pinned to supervisor would scan
    # the backlog forever without producing a single finding (live 2026-11-13
    # CLI run). The core loop is research — so one agent researches. An
    # explicit role_map {1: ...} wins even against the supervisor guarantee:
    # with no swarm left to arbitrate, the owner's pick IS the right answer.
    if n_agents == 1:
        if 1 in role_map:
            try:
                return [Role(str(role_map[1]).lower())]
            except ValueError:
                pass                      # bad entry: use the default below
        return [Role.RESEARCHER]
    roles: list[Role] = []
    for i in range(1, n_agents + 1):
        raw = role_map.get(i, "")
        try:
            roles.append(Role(str(raw).lower()))
        except ValueError:
            roles.append(ALL_ROLES[(i - 1) % len(ALL_ROLES)])
    # guarantee at least one supervisor and one integrator for the machinery.
    # The integrator guarantee needs a swarm big enough to PRODUCE work: at
    # n=2 an integrator has no coder and no MR to drain, and flipping the
    # second agent to integrator would leave the lab with no researcher at
    # all (same degenerate outcome as the lone-supervisor bug above).
    if Role.SUPERVISOR not in roles and n_agents > 0:
        roles[0] = Role.SUPERVISOR
    if Role.INTEGRATOR not in roles and n_agents > 2:
        roles[-1] = Role.INTEGRATOR
    return roles
