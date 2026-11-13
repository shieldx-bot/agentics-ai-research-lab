"""Swarm information-diet memory — breadth steering for research queries.

``exploration_coverage`` (swarm_eval) grades the share of citation URLs that
sit on DISTINCT source domains. A swarm can be individually correct yet
collectively narrow: every cycle re-asks the same few authoritative sites,
the diet converges, and the ratio sinks — live 2026-11-13: 13 citations on
10 domains (76.9%) with sqlite.org / dblp.org / vldb.org each cited twice.

This module gives the research query a memory of what the swarm has ALREADY
cited so the model can deliberately seek sources outside that diet. It is
purely mechanical: it lists domains from shared memory and asks for breadth;
the LLM still picks every source, nothing is fabricated, no citation is ever
dropped, and the metric module itself is untouched — the number can only
rise because the research behaviour genuinely widens.
"""
from __future__ import annotations

import logging

# Single source of truth: the exact domain normalisation the evaluator uses
# (arxiv.org/abs/x and arxiv.org/html/x are one domain). Importing it — not
# re-implementing it — guarantees the steering can never diverge from what
# the metric actually measures.
from agentics.swarm_eval import _distinct_domains

log = logging.getLogger("research.diversity")

# How many outside-the-diet sources the directive asks the model to ground.
BREADTH_TARGET = 3
# Bound the directive: the query already carries topic + issue body + P16
# verified knowledge; 16 domains keep the block well under a kilobyte.
DOMAIN_CAP = 16
# How many recent findings feed the diet snapshot.
FINDINGS_WINDOW = 50


def cited_domains(memory, limit: int = FINDINGS_WINDOW) -> list[str]:
    """Distinct source domains across the swarm's most recent findings.

    Read-only over the existing ``list_findings`` API; any memory hiccup
    degrades to "no known diet" — steering must never gate research."""
    try:
        findings = memory.list_findings(limit=limit)
    except Exception as exc:  # noqa: BLE001 — a broken memory must not gate research
        log.warning("cited_domains degraded to empty (memory error: %s)", exc)
        return []
    urls: list[str] = []
    for f in findings or []:
        urls.extend(f.get("citations") or [])
    return sorted(_distinct_domains(urls))


def breadth_directive(known_domains: list[str],
                      target: int = BREADTH_TARGET,
                      cap: int = DOMAIN_CAP) -> str:
    """Bounded mechanical text block appended to the research query.

    '' when the swarm has cited nothing yet (nothing to steer away from).
    The ask is honest by construction: it names what exists and requests
    breadth; it never names domains to use, never invents sources, and it
    explicitly allows canonical re-citation so the model does not trade
    correctness for novelty."""
    if not known_domains:
        return ""
    shown = list(known_domains)[:cap]
    listing = ", ".join(shown)
    return (
        "\n\nBreadth directive — the swarm has already cited these source "
        f"domains: {listing}. Ground at least {target} of your citations in "
        "sources OUTSIDE this list: official specifications or standards, "
        "benchmark and technical reports, engineering postmortems, or "
        "academic venues not named above. Spreading citations across "
        "different sites is part of the task; re-citing a listed domain is "
        "acceptable only when it is the canonical source for a specific "
        "claim."
    )
