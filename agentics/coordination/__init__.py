"""Coordination pack A2X — cross-agent coordination upgrades.

Three mechanical (zero extra LLM cost) mechanisms distilled from the
multi-agent literature and famous open-source frameworks; each closes one
measured coordination gap of the lab (see docs/COORDINATION.md):

* ``allocation`` — Contract-Net-lite: the supervisor scores every open task
  against every live claimant and awards the best pairings (Contract Net
  Protocol's award step; CrewAI's hierarchical manager; Anthropic's
  orchestrator-worker).
* ``dossier`` — the handoff pack for the verification tier: the reviewer
  sees a task's FULL history (earlier attempts, prior objections) before
  judging an MR (Cognition's "share full traces" + communication bridge;
  MetaGPT's SOP artifact handoffs; MAST inter-agent misalignment).
* ``lessons`` — swarm-wide failure patterns aggregated into short signatures
  with repeat counts, injected at plan time (Voyager's lesson loop;
  Reflexion; GEA group experience sharing).

Everything here is deterministic, offline-testable, and must never break a
cycle: callers wrap these helpers the same way every other memory read in
the graph is wrapped.
"""
from agentics.coordination.allocation import recommend_awards
from agentics.coordination.dossier import review_dossier
from agentics.coordination.lessons import swarm_lessons

__all__ = ["recommend_awards", "review_dossier", "swarm_lessons"]
