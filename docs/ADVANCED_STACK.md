# Advanced Stack — Operations & KPIs

The five advanced technologies integrated on 2026-11-14 (commit `04ebaac`),
how to enable each one, how they degrade when their dependencies are absent,
and the SQL/KPI queries that tell you whether they are actually working
after a few hours of `python -m agentics.cli run`.

All queries below assume the shared memory database at `data/shared_memory.db`
(override with `AGENTICS_DATA_DIR`; every snippet is read-only).

| # | Technology | Default | Enable with |
|---|------------|---------|-------------|
| 1 | PUCT discovery frontier + entropic credit | **on** | nothing (tune via `PUCT_C`, `ENTROPIC_BONUS_MAX`) |
| 2 | MAP-Elites archive + islands (`qd_cells`) | **on** | nothing |
| 3 | Local small-model router (Ollama) | **off** | `LOCAL_LLM_ENABLED=1 LOCAL_LLM_MODEL=qwen3:1.7b` |
| 4 | GraphRAG-lite (`kg_entities` / `kg_edges`) | **on** (needs #3 for extraction) | nothing; extraction starts once #3 is enabled |
| 5 | Hypothesis Arena (Elo tournament + pairing) | **on** | nothing (tune via `AGENTICS_TOURNAMENT_*`) |

---

## 1. Quick start — enable the free local edge (Ollama)

```bash
# one-time: pull a small chat model (1.7B runs on CPU, ~1.4 GB)
ollama pull qwen3:1.7b

# verify the runtime answers
curl -s http://127.0.0.1:11434/api/tags | head -c 300

# run the lab with the local edge on
LOCAL_LLM_ENABLED=1 LOCAL_LLM_MODEL=qwen3:1.7b python -m agentics.cli run
```

Environment knobs (all read from `agentics/config.py`):

| Env | Default | Meaning |
|-----|---------|---------|
| `LOCAL_LLM_ENABLED` | `0` | master switch; off = no localhost calls at all |
| `LOCAL_LLM_URL` | `http://127.0.0.1:11434` | OpenAI-compatible/Ollama base URL |
| `LOCAL_LLM_MODEL` | *(empty)* | model tag, e.g. `qwen3:1.7b`, `qwen3:4b` |
| `LOCAL_LLM_TIMEOUT` | `20.0` | per-request timeout, seconds |
| `LOCAL_LLM_PURPOSES` | `kg_extract,plan,role_synthesis` | comma-separated purposes allowed to route local |

Two routing behaviors (in `agentics/graph/nodes.py:_llm_ask` and
`agentics/research/kg_extract.py`):

1. **Primary routing** — `kg_extract` (the GraphRAG mining extraction)
   calls the local model directly, every time. Zero cookie cost.
2. **Budget fallback (P20c)** — when the P20 pool refuses a purpose that
   is allowed to route local, the ask retries once against the local
   model instead of dying. Default set: `kg_extract`, plus the two
   SCAFFOLDING purposes `plan` (mission synthesis) and `role_synthesis`
   (system-prompt synthesis) — a dry pool now degrades to a local plan,
   not an idle agent (live 2026-11-14 08:35: every cycle starved at
   `plan blocked by budget`). Journal event: `local_fallback` with
   `{"purpose": ...}`.

Hard guarantees (see `agentics/llm/local_router.py` docstring):

* Science purposes (`research`, `verify`, `ideation`, `implement`,
  `review`, `repair`, `defend`, `tournament`) are **never** routed to the
  small model, even if someone adds them to `LOCAL_LLM_PURPOSES` —
  `allowed_for()` refuses them by construction. A weaker verdict is worse
  than a refused ask.
* The router never raises: a missing/stopped Ollama degrades to the old
  behavior (budget refusal / skipped extraction) with a `log.warning`,
  never a crash.
* Local asks do **not** count against the P20 ledger (`llm_usage` only
  records pool calls) — the quota stays clean for the cookie pool.

## 2. What each technology does (one paragraph each)

**#1 PUCT frontier + entropic credit** (`discovery_frontier_puct`,
`_branch_feedback`): the deep-dive backlog is ordered by
`score = Q + c * scale * P * sqrt(1+T) / (1 + n)` — Q is the MAX reward of
settled descendants (best outcome reachable *from* a branch), P is the
rank prior over the discovery buffer, n is settles in the subtree so the
exploitation bonus decays as a branch is dug into. Replaces plain BFS; the
structural cure for "rich-get-richer" discovery chasing.

**#2 MAP-Elites + islands** (`qd_cells` table, `qd_cells_list`): every
qualified hypothesis is placed in a `mechanism|angle` cell; only the best
quality per cell is kept, bounded to the top cells. Islands (round-robin
assignment) keep lineages from collapsing into one basin. PUCT manages
depth, MAP-Elites manages breadth — together they attack the
"topic already covered → skip" saturation mode.

**#3 Local router**: see above. It is the operational keystone — it
converts the whole class of "121/120 pool calls, plumbing starved the
science paths" incidents (2026-11-14 07:08) into a no-op.

**#4 GraphRAG-lite** (`agentics/research/kg_extract.py`, `kg_entities`,
`kg_edges`, `kg_context`): findings and settled claims are mined into a
typed entity–edge graph (method/dataset/metric/concept nodes; improves /
uses / measures / ... edges). Research and ideation prompts get graph PATH
context instead of a flat finding list; novelty checks upgrade from
text-dup to path-dup. Extraction runs on the local router (`kg_extract`
purpose) — without #3 configured, ingest is a no-op.

**#5 Hypothesis Arena** (`agentics/coscientist/tournament.py`, `elo.py`,
`pairing.py`, tables `tournament_matches` / `tournament_feedback`,
`hypotheses.elo` column): pairwise independent reviews produce Elo ratings;
pairing prefers close-Elo opponents (debate mode) and uses vector
similarity for diversity when available. Elo is only a **prior** —
settle/verify with real code execution remains the ground truth.

## 3. KPI queries — run after a few hours

```bash
sqlite3 data/shared_memory.db
```

**K3 — is the local edge earning its keep?** (fallback saves per hour)

```sql
SELECT COUNT(*) AS local_fallbacks,
       COUNT(DISTINCT agent) AS agents_rescued
FROM journal
WHERE event='local_fallback' AND ts > strftime('%s','now') - 14400;
```

**K-budget — did pool exhaustion disappear?** (`budget_blocked` should drop
to near zero for cheap purposes; science purposes still blocking is what
the P20b reserve is for — see `POLICIES.md`)

```sql
SELECT purpose, COUNT(*) FROM journal
WHERE event='budget_blocked' AND ts > strftime('%s','now') - 14400
GROUP BY purpose;
```

**K4 — knowledge graph health** (grows only while the local router serves
`kg_extract`; flat = Ollama is down or the model errors)

```sql
SELECT (SELECT COUNT(*) FROM kg_entities) AS entities,
       (SELECT COUNT(*) FROM kg_edges)     AS edges,
       (SELECT MAX(ts) FROM kg_edges)      AS last_edge_ts;
```

**K2 — archive coverage** (cells and islands in play; coverage growth
without quality decay = the breadth lever is working)

```sql
SELECT COUNT(*) AS cells,
       COUNT(DISTINCT island) AS islands,
       ROUND(AVG(best_quality), 3) AS avg_q,
       ROUND(MAX(best_quality), 3) AS best_q
FROM qd_cells;
```

**K1 — PUCT is digging where the reward is** (are deep-dives anchored in
discovery descent lines?)

```sql
SELECT COUNT(*) AS frontier_branches,
       SUM(CASE WHEN protocol_code != '' THEN 1 ELSE 0 END) AS with_protocol
FROM hypotheses
WHERE status='proposed' AND depth <= 4;
```

**K5 — arena activity** (matches decided and Elo spread over settled
hypotheses; spread ≈ 0 means the arena hasn't kicked in yet)

```sql
SELECT COUNT(*) AS matches,
       SUM(CASE WHEN winner != '' THEN 1 ELSE 0 END) AS decided
FROM tournament_matches;

SELECT MIN(elo), MAX(elo), COUNT(*) FROM hypotheses
WHERE elo != 1200;
```

**End-to-end KPI (the headline number)** — hours from lab start to the
first detected/confirmed discovery; the TTT-Discover target was ≥30%
improvement over the pre-integration baseline. Journal events:
`discovery_detected` (novelty gate passed) and `discovery_confirmed`
(second judge agreed). Per-run first-confirmation time:

```sql
SELECT MIN(ts) - (SELECT MIN(ts) FROM journal) AS seconds_to_first
FROM journal WHERE event='discovery_confirmed';
```

## 4. Ops notes and failure modes

* **Ollama down / model missing** → router returns `ok=False`, lab logs a
  warning and continues exactly as before. Nothing to clean up; fix with
  `ollama serve` / `ollama pull`.
* **`qwen3:1.7b` vs `qwen3:4b`**: 1.7B is fine for `kg_extract` and
  fallback plumbing on CPU; 4B gives better extraction recall if you have
  ~3 GB RAM spare. Both are far below the science bar — which is exactly
  why science purposes can never route there.
* **Do not widen `LOCAL_LLM_PURPOSES` beyond plumbing/scaffolding
  purposes** (extract, plan, prompt synthesis, classify). The router
  enforces the science blacklist regardless; listing a science purpose
  would just be ignored.
* **Health check is GET, not POST**: Ollama's `/api/tags` answers 405
  Method Not Allowed to a POST. Before the fix (live 08:35) the lab
  logged `local_router .../api/tags -> HTTP 405` every cycle while
  Ollama was actually running, and kg ingest silently mined nothing.
  If that line ever reappears, the URL points at a runtime without an
  Ollama-style `/api/tags` (llama.cpp/LM Studio: use their OpenAI-
  compatible health endpoint instead).
* **PUCT tuning**: `PUCT_C` (default 1.0) trades exploration vs reward;
  raise it if the frontier keeps picking the same discovery line.
  `ENTROPIC_BONUS_MAX` (default 8.0) caps the entropy credit from
  `_branch_feedback`; `REWARD_SCALE` (`target` default) selects the Q
  normalization.
* **Tables are bounded by design**: `qd_cells` prunes to the top cells,
  `kg_entities` dedups by name, `llm_usage` rolls its window — no
  unbounded growth from this stack.
* **Integrity**: pull/merge re-anchors the manifest automatically
  (`integrity_reanchored` journal event); a drift *without* a HEAD change
  is still a rogue-edit alarm.
