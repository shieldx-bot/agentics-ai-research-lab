"""Role assignment (agentics/agents/roles.py).

Pins the machinery guarantees (supervisor/integrator) and the single-agent
degenerate case: a lone agent must RESEARCH, not supervise — the live
2026-11-13 CLI run started one agent (one Hyperbrowser token) as supervisor
and scanned the backlog forever without ever calling the LLM.
"""
from agentics.agents.roles import Role, assign_roles


def test_lone_agent_researches_not_supervises():
    """One agent + no explicit role_map -> researcher (the lab's core loop);
    the supervisor-only lone scan was the live CLI instability report."""
    assert assign_roles(1, {}) == [Role.RESEARCHER]
    assert assign_roles(1, None) == [Role.RESEARCHER]


def test_lone_agent_explicit_role_map_wins():
    """The owner can still pin index 1 explicitly — the override beats even
    the supervisor guarantee (a lone-agent lab has nothing to arbitrate)."""
    assert assign_roles(1, {1: "supervisor"}) == [Role.SUPERVISOR]
    assert assign_roles(1, {1: "coder"}) == [Role.CODER]
    assert assign_roles(1, {1: "wizard"}) == [Role.RESEARCHER]  # bad entry


def test_two_agents_get_supervisor_and_researcher():
    assert assign_roles(2, {}) == [Role.SUPERVISOR, Role.RESEARCHER]


def test_round_robin_guarantees_supervisor_and_integrator():
    roles = assign_roles(3, {})
    assert roles[0] == Role.SUPERVISOR
    assert roles[-1] == Role.INTEGRATOR
    assert Role.RESEARCHER in roles


def test_machinery_guarantees_apply_over_partial_maps_for_n_ge_3():
    """Predecessor semantics (kept): from three agents up, the integrator
    guarantee wins over a partial explicit map — only the lone-agent case
    lets the owner's pick stand absolute."""
    role_map = {1: "supervisor", 2: "researcher", 3: "coder"}
    assert assign_roles(3, role_map) == [
        Role.SUPERVISOR, Role.RESEARCHER, Role.INTEGRATOR]


def test_invalid_map_entries_fall_back_to_round_robin():
    roles = assign_roles(2, {1: "wizard", 2: "researcher"})
    assert roles == [Role.SUPERVISOR, Role.RESEARCHER]
