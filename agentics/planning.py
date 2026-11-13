"""Adaptive planning & role prompts — the LLM authors its own instructions.

Two hardcoded layers used to sit between the swarm and the model:

1. role system prompts (ROLE_MISSIONS + a fixed template): every agent of a
   role got the same static mission, no matter what the swarm had just been
   through;
2. the planning prompt: a fixed template whose JSON schema the model merely
   filled in (approach, alternatives, risks were not even asked for).

This module replaces both with LLM-authored, experience-conditioned text:

- Role prompts — `resolve_system_prompt` returns a per-(role, category)
  mission WRITTEN by the LLM (one synthesis call per swarm per TTL window),
  cached in the shared ``role_prompts`` table, reworked after repeated
  failures and conditioned on the swarm's recent traces. The immutable
  guardrails (``roles.GUARDRAILS``) are re-appended by code after every
  synthesis: the model may rewrite the mission, never the rules.
- Plans — `normalize_plan` is the only fixed part of planning: a minimal
  machine contract (fields the lab consumes mechanically — P13 scope locks,
  P16 empirical verification, the MR body) extracted tolerantly from the
  model's own proposal. Everything the model adds beyond the contract
  (``approach``, risks, alternatives, any extra field) is preserved and
  surfaced to the owner instead of being dropped by a rigid schema.

Every adaptive path degrades to the previous static behavior on failure (no
LLM account, synthesis refused, cache error) — a broken provider turns the
adaptive lab back into the pre-adaptive one, never into a crash.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from agentics.agents.roles import (
    GUARDRAILS,
    Role,
    compose_system_prompt,
    static_mission,
)

log = logging.getLogger("planning")

# Synthesis results shorter than this are treated as refusals / junk answers.
MIN_MISSION_CHARS = 40


# ------------------------------------------------------------ role prompts
SYNTHESIS_META = (
    "You are the prompt architect of an autonomous AI research lab. Write the "
    "operating instructions (the mission) for ONE agent.\n"
    "Role: {role}. Task category: {category}. Research domain: {topic}\n\n"
    "What this role has been through lately (learn from it — keep what worked, "
    "address what failed):\n{experience}\n\n"
    "{record_line}"
    "Write 3-8 concrete, actionable sentences in second person ('You ...'): "
    "the mission, how to work well in this category, heuristics that worked "
    "for this swarm, pitfalls to avoid (from the failures above). Plain text "
    "only — no markdown, no preamble, no surrounding quotes. Hard limit: "
    "{max_chars} characters."
)

RECORD_LINE = (
    "The mission currently in use has run {uses} cycle(s): {successes} ended "
    "well, {failures} failed — rewrite it to keep what worked and fix what "
    "did not.\n\n"
)


def sanitize_mission(text: str, max_chars: int = 1600) -> str:
    """Make an LLM-authored mission safe to embed in a system prompt.

    Strips control characters (except newlines/tabs), collapses blank-line
    runs, trims and caps the length. The text is data, never code: it is
    embedded verbatim into the mission slot of the identity template and the
    guardrails are appended afterwards by `compose_system_prompt`."""
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "…"
    return cleaned


def _settings_flag(settings: object, name: str, default):
    return getattr(settings, name, default)


def _role_val(role: object) -> str:
    return getattr(role, "value", None) or str(role or "researcher")


async def synthesize_mission(
    ctx, role: str, category: str, *, row: dict | None = None,
) -> str | None:
    """One LLM call that WRITES the mission for (role, category).

    The call goes directly through ``ctx.llm.ask`` with the static prompt as
    its system message — never through ``_llm_ask`` — so synthesis cannot
    recurse into itself. Returns None (caller falls back to the static
    mission) when the provider fails or the answer looks like a refusal."""
    memory = ctx.memory
    settings = getattr(ctx, "settings", None)
    topic = _settings_flag(settings, "research_topic", "")
    max_chars = int(_settings_flag(settings, "role_prompt_max_chars", 1600) or 1600)
    experience = memory.role_experience(role, category)
    record_line = ""
    if row and (row.get("uses") or row.get("failures")):
        record_line = RECORD_LINE.format(
            uses=row.get("uses", 0), successes=row.get("successes", 0),
            failures=row.get("failures", 0))
    meta = SYNTHESIS_META.format(
        role=role, category=category, topic=topic,
        experience=experience or "(no recorded experience yet)",
        record_line=record_line, max_chars=max_chars,
    )
    fallback_system = compose_system_prompt(
        getattr(ctx, "agent_id", "agent"), role, topic, static_mission(role))
    resp = await ctx.llm.ask(
        meta, mode="concise", purpose="role_synthesis", system=fallback_system)
    mission = sanitize_mission(getattr(resp, "answer", "") or "", max_chars)
    if len(mission) < MIN_MISSION_CHARS:
        log.info("synthesized mission for %s/%s rejected (too short)", role, category)
        return None
    memory.store_role_prompt(role, category, mission, reason=_reason_for(row))
    log.info("synthesized a new mission for %s/%s (%d chars, reason=%s)",
             role, category, len(mission), _reason_for(row))
    return mission


def _reason_for(row: dict | None) -> str:
    """Why a new mission is being written (stored for the owner's audit)."""
    if not row:
        return "cold_start"
    if row.get("status") == "superseded" or row.get("reason") == "too_many_failures":
        return "rework"
    return "ttl_refresh"


async def _mission_for(ctx, role: str, category: str) -> str | None:
    """Cached-mission lookup + TTL + on-miss synthesis (shared swarm-wide).

    The cache path reads the ACTIVE row; the synthesis path receives the
    LATEST row (active or superseded) so the rework sees how the previous
    mission actually performed."""
    memory = ctx.memory
    settings = getattr(ctx, "settings", None)
    ttl_h = float(_settings_flag(settings, "role_prompt_ttl_hours", 24.0) or 0)
    active = memory.get_role_prompt(role, category)
    if active is not None:
        fresh = ttl_h <= 0 or (time.time() - active["ts"]) < ttl_h * 3600
        if fresh:
            return active["mission"]
    prior = memory.latest_role_prompt(role, category)
    return await synthesize_mission(ctx, role, category, row=prior)


async def resolve_system_prompt(ctx, state: dict | None = None) -> str:
    """The system prompt for this cycle — LLM-authored when possible.

    Resolved at most once per cycle (cached on ``ctx.meta``): the first LLM
    node pays the synthesis when the cache is cold, every later call (and
    every other agent of the same role/category while the TTL lives) reuses
    the stored mission. Any failure falls back to the static prompt."""
    state = state or {}
    role = _role_val(getattr(ctx, "role", None))
    category = str(state.get("category") or role)
    meta = getattr(ctx, "meta", None)
    if isinstance(meta, dict) and meta.get("_system_prompt"):
        return meta["_system_prompt"]
    settings = getattr(ctx, "settings", None)
    enabled = _settings_flag(settings, "adaptive_prompts", True)
    topic = _settings_flag(settings, "research_topic", "")
    mission: str | None = None
    if enabled and getattr(ctx, "llm", None) is not None:
        try:
            mission = await _mission_for(ctx, role, category)
        except Exception as exc:  # noqa: BLE001 — adaptive must never break a cycle
            log.warning("role prompt synthesis failed for %s/%s: %s",
                        role, category, exc)
    if not mission:
        mission = static_mission(role)
    prompt = compose_system_prompt(
        getattr(ctx, "agent_id", "agent"), role, topic, mission)
    if isinstance(meta, dict):
        meta["_system_prompt"] = prompt
    return prompt


def bump_prompt_feedback(ctx, state: dict) -> str:
    """Feed one finished cycle back into the active mission (from reflect).

    error -> failure; yield -> neutral (a lost race is not the mission's
    fault); otherwise -> success. Returns the row status after the bump."""
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None or not _settings_flag(settings, "adaptive_prompts", True):
        return ""
    if state.get("yield_reason"):
        return ""                      # neutral: races are not prompt failures
    role = _role_val(getattr(ctx, "role", None))
    category = str(state.get("category") or role)
    rework_after = int(
        _settings_flag(settings, "role_prompt_rework_failures", 2) or 2)
    try:
        return memory.bump_role_prompt(
            role, category, success=not state.get("error"),
            rework_after=rework_after)
    except Exception as exc:  # noqa: BLE001 — feedback must not kill reflect
        log.debug("role prompt feedback failed: %s", exc)
        return ""


# ------------------------------------------------------------ plan proposal
# The ONLY fixed part of planning: fields the lab consumes mechanically.
# Everything else in the model's proposal is preserved (extras).
PLAN_CONTRACT = (
    '- "files": [{"path": "...", "change": "new|edit"}] — P13 locks this '
    "scope across the swarm; list only files you will actually touch\n"
    '- "steps": ["..."] — the owner watches this as your declared intent\n'
    '- "commit_message": "..."\n'
    '- "verify_cmds": ["python -m pytest ... or python <workspace .py script> '
    'that CHECKS the idea; exit 0 must mean the hypothesis holds"] — P16 '
    "verifies your idea by RUNNING these instead of trusting an LLM's "
    "opinion; declare at least one check for a code task\n"
    '- "hypotheses" (optional, max 2): [{"hypothesis_id": "slug", "claim": '
    '"one testable claim", "prediction": "what the measurement will show", '
    '"metric": "snake_case_metric", "comparison": ">|>=|<|<=|==", '
    '"threshold": 0.5, "code": "<FULL python source: computes the metric, '
    'then prints EXACTLY one final line AGENTICS_RESULT: {\\"<metric>\\": '
    '<number>} (standard library only, deterministic, no network)>"}] — a '
    "pre-registered criterion: the protocol runs twice and the reviewer "
    "re-adjudicates the measurement, so a lying script cannot pass\n"
    '- "reuses": ["tool-name"] — names of registered tools you will reuse '
    "(optional, may be empty)"
)

PLAN_INSTRUCTION = (
    "You are the engineer of this task — PROPOSE the plan yourself; do not "
    "wait for a template. Structure your answer as ONE JSON object with:\n"
    '1. "approach" (required, first): your proposal in your own words — what '
    "you will do and why, alternatives you considered and rejected, the "
    "risks you take, and what result would count as success. This is what a "
    "human reviewer reads first.\n"
    "2. The machine contract — the only fixed fields, consumed mechanically "
    "by the lab (scope locks P13, empirical verification P16, the MR body):\n"
    f"{PLAN_CONTRACT}\n"
    "3. Anything else worth recording (risks, milestones, test data, open "
    "questions, fallback plans) — as extra JSON fields of your own naming; "
    "they are preserved in the plan record and shown to the owner.\n"
    "Reply ONLY with the JSON object."
)

# Defensive caps (prompt injection / runaway answers), not structure.
PLAN_CAPS = {
    "verify_cmds": 5,
    "hypotheses": 2,
    "files": 15,
    "reuses": 10,
}
APPROACH_MAX = 1200
EXTRAS_MAX = 2000
_PLAN_KEYS = ("files", "steps", "commit_message", "verify_cmds",
              "hypotheses", "reuses", "reusable_tools", "approach")


@dataclass
class PlanCore:
    """What the pipeline needs from the model's proposal — nothing more."""
    parsed: dict = field(default_factory=dict)          # full proposal (extras kept)
    approach: str = ""                                  # the model's own words
    extras: dict = field(default_factory=dict)          # fields beyond the contract
    commit_message: str = "update"
    verify_cmds: list[str] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    reuses: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def normalize_plan(parsed: dict, caps: dict | None = None) -> PlanCore:
    """Tolerant adapter from the model's own plan to the machine contract.

    The model proposes the structure; this extracts only what the machinery
    consumes, with defensive caps, and keeps everything else (approach and
    extra fields) intact for observability. Backward compatible: a proposal
    without verify_cmds/hypotheses behaves exactly like the old plans."""
    caps = {**PLAN_CAPS, **(caps or {})}
    core = PlanCore()
    if not isinstance(parsed, dict):
        return core
    core.parsed = parsed
    core.approach = str(parsed.get("approach", "") or "")[:APPROACH_MAX]
    core.steps = [str(s) for s in (parsed.get("steps") or [])][:16]
    core.commit_message = str(parsed.get("commit_message") or "update")
    core.verify_cmds = [str(c) for c in (parsed.get("verify_cmds") or [])
                        if str(c).strip()][:caps["verify_cmds"]]
    core.hypotheses = [h for h in (parsed.get("hypotheses") or [])
                       if isinstance(h, dict)][:caps["hypotheses"]]
    core.reuses = [n for n in (parsed.get("reuses")
                   or parsed.get("reusable_tools") or [])
                   if isinstance(n, str)][:caps["reuses"]]
    files = parsed.get("files") or []
    core.expected_files = [
        f.get("path", "") for f in files
        if isinstance(f, dict) and f.get("path")
    ][:caps["files"]]
    core.extras = {
        k: v for k, v in parsed.items()
        if k not in _PLAN_KEYS
    }
    return core
