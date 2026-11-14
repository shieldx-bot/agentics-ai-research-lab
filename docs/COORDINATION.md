# Coordination — A2X coordination pack (research → upgrades)

Owner's complaint (2026-11-13): *"cách các agent phối hợp và làm việc với nhau
đang còn chưa hiệu quả"* — the way the agents collaborate is not effective
yet. This document records the research sweep over the multi-agent
literature and well-known open-source frameworks, the gap analysis against
THIS lab, and the three mechanical upgrades that came out of it.

Every upgrade is deterministic code (zero extra LLM calls), offline-tested,
and wired into the existing LangGraph cycle without changing any policy
(P1–P24) contract.

---

## 1. What the research says

### 1.1 Frameworks / systems (famous source code)

| System | Coordination idea | Relevance to this lab |
| --- | --- | --- |
| **MetaGPT** (`FoundationAgents/MetaGPT`) — *Meta Programming for a Multi-Agent Collaborative Framework* ([arXiv:2308.00352](https://arxiv.org/abs/2308.00352)) | Encode SOPs; every role hands the next role a **structured artifact**, not a chat line ("Code = SOP(Team)") | The lab already hands artifacts (plan JSON, MR records) between nodes; the handoff to the *reviewer* tier was the missing artifact → fixed by the **dossier** |
| **ChatDev** (`OpenBMB/ChatDev`) — dual-agent **Chat Chain**, instructor + builder per phase, [arXiv:2307.07924](https://arxiv.org/abs/2307.07924) | Phase-by-phase two-agent critique | Already implemented natively as the **P19 peer panel** (vote → defense → re-vote), which is stricter (unanimity, mechanical rulebook) |
| **AutoGen** (Microsoft) — group chat, speaker selection | A shared message thread all agents subscribe to | The lab's journal/broadcast bus is the same pattern with DB persistence; no change needed |
| **CrewAI** — **hierarchical process**: a manager agent allocates tasks and validates outcomes ([docs](https://docs.crewai.com/en/learn/hierarchical-process)) | Manager-driven allocation | The lab's supervisor was a janitor, not a manager → fixed by **Contract-Net-lite allocation** |
| **Voyager** (Nvidia/MineDojo) — skill library + **lesson loop**: "store sequences that work and remember what does not" ([voyager.minedojo.org](https://voyager.minedojo.org), [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) | Lifelong learning from aggregated experience | The lab stores skills (LifeMem) and traces, but failures were only ever read row-by-row → fixed by **swarm lessons** (aggregated failure signatures) |
| **Contract Net Protocol** (Smith 1980; classic MAS task allocation) | announce → bid → award | The lab had announce (backlog) and bid (P–N score in `node_claim`) but NO award step → fixed by **allocation** |

### 1.2 Production experience (essays + papers)

* **Anthropic — "How we built our multi-agent research system"** and the
  follow-up research *"Patterns and problems in emerging multiagent systems"*
  ([anthropic.com/research/multiagent-systems](https://www.anthropic.com/research/multiagent-systems)):
  orchestrator-worker with parallel subagents; the LEAD agent decomposes
  work and evaluates results. Lesson taken: a coordinator that only
  cleans up is a wasted orchestration tier.
* **Cognition — "Don't Build Multi-Agents"** and **"Multi-Agents: What's
  Actually Working"** ([cognition.com/blog](https://cognition.com/blog/multi-agents-working)):
  (1) *share as much context as possible — agents must see the same
  sources, the same todo list, the same priors*; (2) *actions carry
  implicit decisions* → keep writes single-threaded; (3) verification
  agents should work from a clean context but receive a proper
  **communication bridge**; (4) the open problems are all
  *communication* problems. Lessons taken: hand the reviewer the full task
  history (dossier), keep the award a market signal instead of a hard
  assignment (no central serialization of writes).
* **MAST — "Why Do Multi-Agent LLM Systems Fail?"** (Berkeley, NeurIPS
  2025, [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)): the first
  failure taxonomy (14 failure modes, 3 categories — system design,
  **inter-agent misalignment**, **task verification**; 150+ traces,
  κ = 0.88). Both categories map directly to the two gaps fixed here:
  reviewers blind to task history (misalignment) and lessons that never
  reach the next attempt (verification).

### 1.3 What the lab ALREADY had (no change needed)

Prior art is already integrated deeper than most frameworks:

* shared blackboard (SQLite WAL journal + ChromaDB vectors) — the
  Hearsay-II pattern; AutoGen-style group thread with persistence;
* P19 **peer panel** — ChatDev critique, but with unanimity + defense
  rounds enforced mechanically (`agentics/peer_review.py`);
* **P–N Performance–Novelty task scoring** (`agentics/evolution/scoring.py`)
  — the claim-time "bid";
* LifeMem skills + role prompts reworked from failures (GEA-style group
  experience, adaptive prompts);
* P23 replication replay, P16 evidence ledger, P13 intent locks
  (single-threaded writes per file — exactly Cognition's rule).

---

## 2. Gap analysis → the three upgrades (A2X)

| # | Gap found in code | Failure it causes | Fix | Prior art |
| --- | --- | --- | --- | --- |
| 1 | `node_scan` (supervisor) = janitor only; nobody routes work | Best-fit agents race the same tasks while tasks they excel at sit unclaimed; allocation quality = race luck | **Contract-Net-lite** — every scan, the supervisor scores every open task × every live claimant and **awards** the best pairings; `node_claim` lifts an awarded task by `coord_award_bonus` (default 0.75) | Contract Net (award step); CrewAI hierarchical manager; Anthropic orchestrator |
| 2 | Reviewer prompt = diff + author's claims + evidence rows; task history invisible | MR re-implements an approach that already failed twice on this task and passes review (MAST inter-agent misalignment) | **Reviewer dossier** — bounded "Task history" block: earlier attempts (agent/outcome/failure) + prior MR objections for the same task | Cognition communication bridge; MetaGPT SOP handoffs |
| 3 | Swarm failures only ever consumed raw (`trace_context`, 3–4 rows) or personally (`_lessons_text`) | The same dead-end approach is retried across agents for cycles (MAST task-verification) | **Swarm lessons** — failures in the plan's category aggregated into signatures with repeat counts `(xN; last: aXX)`, injected at plan time; the planner's own rows stay in the personal-lessons block (no double counting) | Voyager lesson loop; Reflexion; GEA |

## 3. Where it lives

* `agentics/coordination/allocation.py` — `recommend_awards()`,
  `award_digest()`; awards table `task_awards` in the shared DB
  (rewritten wholesale every scan — a stale award cannot survive one
  cycle). Wired in `node_scan` (journal `contract_net_awards` + broadcast)
  and `node_claim` (bonus + `+award` tag in the claim decision).
* `agentics/coordination/dossier.py` — `review_dossier()`; wired into the
  reviewer prompt in `node_review`.
* `agentics/coordination/lessons.py` — `swarm_lessons()`; wired into the
  planner prompt in `node_plan`.
* Config knob: `COORD_AWARD_BONUS` (float, default `0.75`, `0` disables
  the award signal; the table is still maintained so the dashboard/journal
  keep showing what the supervisor would have awarded).

## 4. What was deliberately NOT adopted

* **Dynamic role mutation mid-run** (AgentVerse-style recruitment): role
  changes reshuffle claims, branches and panel eligibility in flight; the
  award signal reaches the same goal (right work → right agent) without
  mutating identity. Revisit only if allocation proves insufficient.
* **Free-form agent chat** (AutoGen group-chat-as-conversation): message
  storms burn the per-token LLM budget; the lab's bus is structured
  (journal events + typed messages), which is what MAST recommends against
  misalignment.
* **Hard assignment** (manager serializes writes): against Cognition's
  "single-threaded writes" only in spirit — here each agent still owns its
  decision and race; the manager informs, it does not lock.
* **A2A/MCP network protocols**: single-process, single-machine lab; the
  shared DB already is the protocol.

## 5. Validation

* `tests/test_coordination_allocation.py` — pairing correctness, one-per-
  agent determinism, coder-only evolution, table lifecycle, node_scan
  journal/broadcast, node_claim flip + no-award invariance.
* `tests/test_coordination_dossier.py` — attempts + objections content,
  MR-under-review boundary, fresh-task emptiness, bounding, prompt wiring
  (present with history, absent without).
* `tests/test_coordination_lessons.py` — aggregation with xN/last-agent,
  self-exclusion, rejected-MR inclusion, empty + bounded edges, plan-prompt
  wiring.

All 819 tests green (`pytest tests/ -q`) at the commit that introduced the
pack.
