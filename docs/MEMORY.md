# Memory — shared, personal, and LifeMem structuring

The lab's memory has three layers. The first two existed since v1 (GEA
generation); the third is the **LifeMem upgrade** (Qiu et al.,
[arXiv:2609.12655](https://arxiv.org/abs/2609.12655)) that stops the shared
memory from degrading into an unorganized pile as the lab runs for weeks.

## Layer 1 — Shared memory (`agentics/memory/shared_memory.py`)

SQLite WAL, one file (`data/memory.db`), safe for concurrent agent threads.

| Table | Purpose |
| --- | --- |
| `messages` | broadcasts, handoffs, alerts between agents |
| `findings` | research results with citations |
| `decisions` | durable who-did-what-and-why log |
| `claims` | atomic issue-claim registry (P1) |
| `heartbeats` | agent liveness (P6) |
| `traces` | GEA evolutionary traces: what was tried, what failed, outcome |
| `tool_registry` | reusable tools discovered across agents (GEA) |
| `merge_queue` | persistent serial merge queue (P4) |
| `trace_clusters` | **LifeMem**: workflow clusters grouping similar traces |
| `skills` | **LifeMem**: skills distilled from merged traces per cluster |

## Layer 2 — Personal memory (`agentics/memory/personal_memory.py`)

Per-agent SQLite (`data/personal/<agent>.db`): lessons this agent drew from its
own cycles. Cheap, private, always local to the agent's context.

## Layer 3 — LifeMem structuring (`agentics/memory/lifemem.py`)

### The problem it solves

`traces` grows monotonically. Retrieval (`recent_traces`) is recency-based, so
after a few hundred cycles: (1) relevant old experience is buried, (2) nothing
is *abstracted*, so a new agent has to re-learn a workflow from raw traces —
exactly the "unorganized memory" failure mode LifeMem documents for agents.

### What was adopted from the paper

1. **Workflow clustering** — every trace written by `node_reflect` is filed
   into a workflow cluster: same issue *category*, geometrically similar
   `attempted` text. Similar → join; dissimilar → new cluster. Assignment is
   idempotent (an already-clustered trace is never re-filed).
2. **MRL diversity constraint** (Eq. 3 of the paper) — a cluster splits into
   two sub-clusters **only if** it is large (> 50 traces) *and* geometrically
   loose (Mean Resultant Length < 0.95). Homogeneous clusters (repeating
   templates) never split; this is what keeps skill noise low.
3. **Skill distillation** — only *merged* (verified-successful) traces count
   as evidence of a working workflow. When the Integrator merges a PR,
   `on_merge()` re-files its traces, checks for splits, and re-distills the
   cluster's active skill: "Workflow (category): … Avoid: …" (the most recent
   failure becomes the avoid-clause). A new skill supersedes the old one.
4. **Dual-tier recall** — `node_plan` receives BOTH concrete traces
   (`trace_context`, env-aligned exemplars) AND distilled skills
   (`skill_context`, transferable abstractions), mirroring the paper's
   M_act = (trajs, skills).

### Deliberate lab adaptations (not in the paper)

- **Embeddings**: deterministic hashed bag-of-words (64-d, `zlib.crc32`),
  no model download, no network — the whole module runs and tests offline.
  Sufficient for short plan-digest texts; swap for MiniLM later if needed.
- **Category-first scoping**: clusters only admit traces of the same issue
  category (`research/code/docs/…`) — our coarse analogue of "environment",
  which already prevents most cross-domain retrieval interference.
- **Distillation trigger**: on merge (outcome verified), not on trace write —
  skills are never distilled from unverified work. Offline the distiller is a
  deterministic heuristic; `distill_skill(..., llm_ask=...)` accepts an LLM for
  higher-quality skills.
- **Volume/tau defaults** follow the paper (L=50, tau_mrl=0.95) but are
  function parameters, so tests exercise them small.

### Operating it

```bash
python -m agentics.cli memory-stats   # traces, clusters, active/superseded skills
```

`Top skills` shows the most-activated distilled skills — if a skill's `uses`
count grows while the merge rate stays flat, the skill tier is doing its job.

### Limits

- Hashed BoW misses synonymy ("test suite" vs "tests"); the JOIN_THRESHOLD
  (0.35) is intentionally low, and category scoping bounds the damage.
- Skills are only as good as trace digests; rich `_plan_digest` input (steps,
  not just commit messages) directly improves skill quality.
- Skills are advisory context for the planner, never executed — a bad skill
  can waste a cycle but cannot break a merge (tier-1/tier-2 gates still apply).
