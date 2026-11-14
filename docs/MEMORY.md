# Memory — shared, personal, and LifeMem structuring

The lab's memory has three layers. The first two existed since v1 (GEA
generation); the third is the **LifeMem upgrade** (Qiu et al.,
[arXiv:2609.12655](https://arxiv.org/abs/2609.12655)) that stops the shared
memory from degrading into an unorganized pile as the lab runs for weeks.

A later hardening pass (v2) closed the loops that were half-built — dedup,
recall, bounded growth — without introducing any new subsystem; every
change upgrades a mechanism that already existed. See "v2 hardening" below.
A second pass (v3) tightened the same loops one more notch (vector-tier
sync-trim, offline dedup fallback, number-blind lesson identity, issue-aware
recall) — see "v3 hardening".

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
| `messages` / `findings` / `decisions` | v2: bounded by the same trim policy the journal/evidence feeds already had |

## Layer 2 — Personal memory (`agentics/memory/personal_memory.py`)

Per-agent SQLite (`data/personal/<agent>.db`): lessons this agent drew from its
own cycles. Cheap, private, always local to the agent's context.

v2 closed the write-only gap (lessons used to be written at reflect time and
never read back) and hardened the notebook:

- **Dedup** — a lesson is identified by its normalized hash (the same
  normalization `claim_hash` applies to hypotheses); a recurrence bumps
  `hits` / `last_seen` instead of inserting a duplicate row. Recurring
  failures render as `(xN)` patterns in prompts.
- **Recall** — `recent_lessons` (recency) and `relevant_lessons(query)`
  (keyword overlap + hits, the ranking the tool/skill lookups already use)
  feed `_memory_context["lessons"]` and the `node_plan` prompt.
- **Bounded** — the lessons table self-trims like the journal (every 50 new
  lessons, keep the 500 most recently seen).
- **Migration** — v1 databases upgrade in place on first open (new columns,
  hash backfill, duplicate consolidation); re-opening is idempotent.

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
python -m agentics.cli memory-stats   # traces, clusters, skills,
                                     # findings/messages/decisions/evidence/hypotheses,
                                     # per-agent personal lessons (deduped)
```

`Top skills` shows the most-activated distilled skills — if a skill's `uses`
count grows while the merge rate stays flat, the skill tier is doing its job.

### Reliability fixes

- Cluster splitting must keep the filtered trace records and their 2-means labels
  in the same order; empty-attempt traces remain attached to their original
  cluster. After a split, each resulting cluster is re-distilled independently.
- The offline hashed BoW implementation remains a baseline, not a semantic
  embedding claim. Evaluate any future embedding replacement against retrieval
  benchmarks before changing the default.

## v2 hardening — closing the half-built loops

Everything below upgrades an existing mechanism; nothing new was invented.

| Loop that was half-built | What the upgrade does |
| --- | --- |
| Personal memory was write-only | `_memory_context["lessons"]` + `node_plan` inject the agent's own deduped lessons (recency snapshot; relevance-ranked at plan time) |
| Findings vector index was dedup-only | `finding_recall_text` queries the SAME ChromaDB index at plan time ("what do we already know?"); hits pointing at trimmed SQLite rows are skipped via `get_finding` — recall never cites a ghost |
| Identical lessons re-inserted every cycle | hash dedup + `hits`/`last_seen` (the `claim_hash` idea, applied to lessons) |
| messages/findings/decisions grew forever | the journal/evidence trim pattern extended to them (10k messages / 2k findings / 2k decisions kept) |
| hot queries had no indexes | `idx_messages_ts`, `idx_findings_issue`, `idx_journal_agent`, `idx_evidence_agent` |
| `memory_stats()` escaped the lock for its count queries | all stats now run under the connection lock (latent cross-thread hazard on the shared conn) |

All recall/injection paths are defensive: a context or double without the
new surface (old fixtures, test fakes, a partially started swarm) degrades to
an empty string — memory can never break a cycle. At cold start every new
prompt block is empty, so prompts are unchanged until memory actually has
something to say.

## v3 hardening — same loops, one notch tighter

Nothing new was invented here either; each row upgrades a mechanism that
already existed (trim, dedup, recall, fallback).

| Half-built loop (v2 left it at) | What the v3 upgrade does |
| --- | --- |
| Findings trim only trimmed SQLite — the ChromaDB tier kept every doc forever (ghosts ate the `n_results` budget and the index grew unbounded) | **Sync-trim**: each finding row stores its `vector_doc_id`; when the trim evicts rows it hands the evicted doc ids to a hook the orchestrator wires to `VectorStore.delete_many` — both tiers stay the same size. `memory-stats` prints the drift (`Vector docs: N`) so the owner can see the tiers track. |
| P7 research dedup needed chromadb — with the vector store offline the gate returned "unknown" every cycle | **SQLite fallback**: `finding_search` (keyword overlap over the newest 500 finding titles) answers the same gate when `vector.available` is False — the fallback pattern the policies layer already used, applied to the dedup gate. A match needs ≥ 2 significant shared words, so one common word never counts as coverage. |
| Lesson dedup was number-sensitive — "cap reached (57/50)" and "(63/50)" sat as two rows, so recurring failures with varying counts never showed `(xN)` | **Number-blind identity**: `lesson_hash` collapses digit runs before hashing (verbatim text stays stored for display); v2 notebooks re-normalize + re-consolidate in place on first open, guarded by a `state` key so the sweep runs exactly once. |
| `relevant_lessons` ignored the `issue` column (written since v1, never read) | **Issue-aware recall**: lessons recorded on the same issue get +3 on the keyword-overlap score, so a retry/rework of issue N finds this agent's own history on N even when the wording shares no keywords. `node_plan` passes `issue_number`; old fakes without the kwarg fall back to the v2 call. |

Operational notes:

- The sync-trim hook runs outside the SQLite lock and swallows its own
  failures — a chroma hiccup can delay ghost cleanup but never breaks the
  finding insert that triggered the trim. Legacy rows written before v3 have
  no `vector_doc_id`; their docs (if any) are cleaned lazily and are already
  skipped at read time by the v2 ghost-check.
- Digit-blind identity is deliberately coarse: "exit code 1" and "exit code 2"
  are one lesson. For recurring-failure detection that is the point — the
  number is the cycle's, the lesson is the swarm's.

### Limits

- Hashed BoW misses synonymy ("test suite" vs "tests"); the JOIN_THRESHOLD
  (0.35) is intentionally low, and category scoping bounds the damage.
- Skills are only as good as trace digests; rich `_plan_digest` input (steps,
  not just commit messages) directly improves skill quality.
- Skills are advisory context for the planner, never executed — a bad skill
  can waste a cycle but cannot break a merge (tier-1/tier-2 gates still apply).

## GraphRAG verdict layer — tin cậy + đa hop + mâu thuẫn (roadmap #4, đã triển khai)

Trên nền `kg_entities`/`kg_edges` (khai thác triple từ findings/settles qua
`research/kg_extract.py`, LOCAL router), `SharedMemory` giờ có lớp phán quyết:

- **Trust provenance (P16)** — cột `kg_edges.trusted`: khi một hypothesis sinh
  ra cạnh đó bị settle `refuted`, mọi cạnh có `source_hyp` = nó được đánh dấu
  `trusted=0` NGAY trong `update_hypothesis` — máy giết triple, đồ thị không
  được quên. `kg_trust_report()` trả edges/trusted/untrusted.
- **Multi-hop retrieval** — `kg_neighborhood(terms, hops≤2)`: BFS từ seed
  (LIKE entity), mỗi hop đánh dấu `h1:`/`h2:`, mặc định chỉ phục vụ cạnh
  trusted; `kg_neighborhood_from_text(claim)` tự seed bằng token khớp entity
  CHÍNH XÁC (không LLM, không fuzz).
- **Propose-time context** — `_gate_and_register` chèn block neighbourhood
  vào JUDGE prompt khi đồ thị có entity liên quan (flag `kg_propose_context`,
  mặc định bật; đồ thị rỗng → prompt byte-đồng-nhất như trước).
- **Mâu thuẫn ngữ nghĩa** — `kg_contradictions(threshold=0.55)`: cặp claim đã
  chốt NGƯỢC phán quyết với chồng từ Jaccard cao nhưng criterion KHÁC nhau —
  lớp mà alarm cùng-criterion (`claim_contradiction`) không thấy.

```bash
agentics kg-query lstm --hops 2        # lái đồ thị (cạnh untrusted có nhãn)
agentics kg-contradictions             # cặp mâu thuẫn + vệ sinh đồ thị
```

Ranh giới trung thực: cạnh CHỈ sinh từ sự kiện có nguồn (kg_extract có
`source_hyp`); đồ thị rỗng = không context, không bịa; KG là context sugar —
không phải stage pipeline, mọi lỗi degrade về không-block.
