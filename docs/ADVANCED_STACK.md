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
| `LOCAL_LLM_TIMEOUT` | `120.0` | per-request timeout, seconds — CPU-realistic: the FIRST call also loads the model into RAM |
| `LOCAL_LLM_NUM_PREDICT` | `512` | bounded generation so CPU inference finishes in seconds |
| `LOCAL_LLM_THINK` | `0` | `1` lets qwen3-family models think (slower; `<think>` blocks are stripped either way) |
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

* **CPU inference is the bottleneck, budget it**: with `qwen3:1.7b` the
  first `/api/chat` also loads 1.3 GB into RAM (30-60s on CPU), the
  model would otherwise think and generate up to 2048 tokens. The router
  therefore sends `num_predict: 512`, `think: false` (strips any leftover
  `<think>` block too) and `keep_alive: 30m` so the model stays warm
  between cycles. Raise `LOCAL_LLM_NUM_PREDICT` only if local answers
  get truncated.
* **Silent `failed: ` warnings are timeouts**: `TimeoutError` stringifies
  to empty — the router now logs `failed: TimeoutError: `. If you see it,
  raise `LOCAL_LLM_TIMEOUT` or lower `LOCAL_LLM_NUM_PREDICT`.
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

---

## 5. Wave 2 — evolution & calibration stack (2026-11-14, commits e00bc29..7319550)

The second five technologies, layered on the wave-1 stack. Shared design
rules: selection signal is EXIT-CODE truth only (never an LLM's opinion);
all generation/mutation/reflection runs on the LOCAL router (zero cookie
cost); every consumer degrades to the previous behavior when its feature
flag is off.

| # | Technology | Default | Env knobs |
|---|------------|---------|-----------|
| 2 | DSPy-lite prompt evolution | on | `PROMPT_EVOLUTION`, `PROMPT_VARIANT_POOL`, `PROMPT_EXPLORE_P`, `PROMPT_VARIANT_PROMOTE_MIN/EDGE` |
| 4 | Verify calibration layer | on | thresholds in `verify/calibration.py` |
| 5 | Reflection tree | on | `REFLECT_EVERY_CYCLES`, `REFLECT_MIN_NOTES` |
| 1 | Skill library | on | `SKILL_LIBRARY`, `SKILL_DISTILL_EVERY_CYCLES`, `SKILL_MAX` |
| 3 | EIG frontier | on (w=0.5) | `EIG_WEIGHT` (0 = pure PUCT) |

**What each does, one line each**

* **#2 Prompt evolution** — the role mission EVOLVES: a bounded pool of
  mutation arms (`prompt_variants`) is served on exploration cycles
  (Thompson sampling, `prompt_explore_p`), scored on real cycle outcomes,
  and promoted through the ordinary role_prompts machinery when it beats
  the champion win-rate by 0.15 with >=10 samples. Mutation via the local
  router (`prompt_evolve`).
* **#4 Verify calibration** — settle notes carry an empirical confidence
  floor `[cal r over n re-runs, Wilson-95 >= lo]` computed from replay +
  replication agreement; the supervisor sweep journals
  `verify_calibration` and alerts ONCE when the floor drops below 0.7
  (n>=10) — because a wrongly confident verdict compounds through PUCT Q,
  Elo and GraphRAG simultaneously.
* **#5 Reflection tree** — every 6 supervisor cycles, recent world notes
  are synthesized into INSIGHT notes (level 1-3, provenance-validated
  parent ids, fabricated citations dropped) on the local router
  (`reflect`); ideation seeds from `[INSIGHT-Ln]` lines in world_render.
* **#1 Skill library** — supported settles are distilled into
  `skill_library` rows (description via the local router `skill_distill`;
  protocol code stored verbatim); retrieval injects the token-overlap
  top-3 into the protocol-writing and implement prompts.
* **#3 EIG frontier** — the PUCT order is refined by information-gain-per-
  cost: Beta-Bernoulli priors per metric from the ledger, closed-form
  mutual information (digamma/Beta entropy), cost = 1 + 0.5*attempts.

**Wave-2 KPI queries** (same `data/shared_memory.db`)

```sql
-- prompt evolution: are challengers winning?
SELECT status, COUNT(*), SUM(uses), SUM(wins) FROM prompt_variants
GROUP BY status;
SELECT reason, COUNT(*) FROM role_prompts WHERE reason='prompt_evolved';

-- calibration: how trustworthy are verdicts this hour?
SELECT rate, wilson_lo FROM (…)  -- or read journal event verify_calibration
SELECT COUNT(*) FROM journal WHERE event='verify_calibration_alert';

-- reflection: insights synthesized and cited by ideation?
SELECT level, COUNT(*) FROM world_notes WHERE kind='insight' GROUP BY level;
SELECT COUNT(*) FROM journal WHERE event='reflection_stored';

-- skill library: compounding or idle?
SELECT COUNT(*), SUM(uses) FROM skill_library WHERE status='active';
SELECT COUNT(*) FROM journal WHERE event='skill_distilled';

-- EIG: is the frontier buying information?
SELECT COUNT(*) FROM journal WHERE event='frontier_priority';
-- compare eig_score of picked vs average proposed claim over time
```

**Failure modes**

* No Ollama → prompt evolution never mutates, no reflections, no skill
  distillation, no kg extraction — everything else runs exactly as before.
  The lab does not NEED the local edge; it just pays full cookie price.
* A promoted variant immediately failing → the existing
  `too_many_failures` rework path supersedes it like any champion; arms
  are not special.
* `EIG_WEIGHT` too high (>1.0) can pull the frontier toward flashy unknown
  metrics over deep-dive continuity — keep it <= 0.5.
