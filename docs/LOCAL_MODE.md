# LOCAL_MODE — the lab runs entirely on one machine (no GitHub)

> Answers the owner's two hypotheses:
>
> **H1 — Can agents work directly in local without corrupting the agent code
> itself?** Yes — enforced by the P10 protected-paths guard (3 layers) plus
> the integrity DETECT layer (layer 4).
>
> **H2 — Can the owner see what the agents are doing, and why (no blackbox)?**
> Yes — every agent publishes its intent at every step; the journal, the
> `status` command and the live web dashboard make it owner-visible.

This document describes the local-only execution mode introduced on this
branch. It supersedes the previous GitHub-coupled workflow: the research cycle
performs **zero network calls** apart from the LLM provider traffic itself.

---

## 1. Architecture

Every GitHub touchpoint of the research cycle has a local replacement:

| GitHub concept         | local replacement                                              |
| ---------------------- | -------------------------------------------------------------- |
| remote repository      | a bare git repo at `data/lab.git` ("origin"), seeded once from the running checkout |
| issues                 | the `tasks` table in shared memory (`data/shared_memory.db`)    |
| claim labels (P1)      | the `claims` table only (single-layer; one process, one lock)   |
| pull requests          | the `reviews` table ("MR" — merge request)                      |
| PR reviews / approvals | verdicts written by Reviewer agents to `reviews`                |
| merge + push to main   | a local `--no-ff` merge pushed as fast-forward to `data/lab.git` (P4) |
| GitHub Actions CI      | tier-1 (compile + fast tests, author) and tier-2 (full suite on main+MR, Integrator) run in the agent workspaces |

The `agentics/github_ops` package stays on disk as **dormant legacy code**
(still covered by its own unit tests) but is no longer imported by the
research cycle. `configs/lab.yaml` now also seeds the local backlog with
starter tasks, and the Supervisor tops the backlog up from a rotation of
survey angles when it runs dry (`agentics/local_ops.py`).

### One research cycle, end to end

```
owner: agentics task "Benchmark X"          # or Supervisor seeds it
  → Supervisor: sweep stale claims, top up backlog
  → Coder/Writer: claim task (SQLite lock, TTL lease)
      publish intent: goal / why / expected files
  → research → plan (LLM JSON: files, steps, reuses)
  → implement: workspace clone of data/lab.git, P10 write gate per file,
      tier-1 (py_compile + pytest), commit, push branch
  → open MR: reviews row (task, branch, files, verification)
  → Reviewer: cross-review (never own MR), P10 re-check,
      verdict approve → merge queue (P4)
  → Integrator: tier-2 (full tests on main+MR), then local merge
      task → merged; tools registered in the plan become active
  → reflect: lesson + trace into shared memory (GEA), journal entry
```

---

## 2. H1 — the agent code cannot be corrupted (5 layers)

**Layer 0 — ownership matrix (P3, `configs/lab.yaml`).** The writable research
surface per role is `experiments/`, `tests/`, `benchmarks/` (coder),
`research_notes/` (researcher), `docs/` (writer), `reports/` (integrator).
The lab's own code is out of bounds for every role.

**Layers 1–3 — protected paths (P10, `agentics/policies/guard.py`).** The
paths below are hard-blocked for ALL agents, at three independent gates:

```
agentics/   configs/   accounts/   scripts/   .github/
```

1. **Write gate** — `RepoWorkspace.write_file` refuses to even create/edit a
   protected file, whatever the plan says.
2. **Diff gate** — `node_implement` scans every changed file before the
   commit; a violation yields the cycle (no commit, no push).
3. **Review + merge gates** — the Reviewer downgrades any "approve" whose
   diff touches protected paths; `LocalMergeQueue.drain` refuses to merge a
   violating MR even if it was approved.

The ONLY exemptions are owner-granted: per task
(`python -m agentics.cli allow-protected <task_id>`, revoke with `--revoke`)
or run-wide (`ALLOW_PROTECTED_PATHS=1`). Credentials (`accounts/`,
`configs/secrets.yaml`, `.env`, `STOP`) are **never** exemptable.

**Supervised evolution lane (follow-up fix).** Before this patch the owner
exemption was dead on arrival: `allow_protected` unblocked only the PUSH
(layer 4), while the write gate (layer 1) refused to even create an
`agentics/` file — an exempt self-evolution task could never produce a
change. The write gate now opens for the agent-code files of an exempt task
(`write_file(allow=...)`, credentials never), while the pre-existing
evolution allowlist still confines WHAT the exempt task may change. Two
credential holes closed in the same pass: `.env` and `STOP` joined
`accounts/` + `configs/secrets.yaml` in `ALWAYS_PROTECTED` (they could ride
an exempt push before), and the `lstrip("./")` normalization bug that made
`_always_protected(".env")` miss was fixed.

**Git layer — pre-receive hook on the origin (`LocalRepo.ensure_guard`,
`origin_hook_script`).** The gates above only see traffic that flows through
agentics code; a raw `git push` to `data/lab.git` bypasses all of them. The
bare origin therefore enforces the same rulebook at receive-pack time:
owner pushes (committer email not `agentics+*`) pass untouched; agents may
never move or delete `main` (work goes through the MR flow); agent commits
are checked PER COMMIT (`git diff-tree`), so a branch rebased on top of fresh
owner commits cannot smuggle changes past the check. The hook is regenerated
from the same protected-path lists the in-process layers read (including the
runtime `PROTECTED_PATHS` / lab.yaml extension) on every boot — do not edit
it by hand; drift is self-healed.

**Layer 4 — DETECT (`agentics/sandbox/integrity.py`).** Layers 1–3 stop agent
*changesets*; nothing yet guarded the tree the orchestrator itself runs from.
`IntegrityGuard` hashes `agentics/`, `configs/`, `scripts/` and the dependency
manifests at startup (`data/integrity/baseline.json`); the Supervisor
re-verifies every 5th cycle; any drift — botched merge, escaped process,
manual edit — becomes an `integrity_alert` journal entry + memory alert.
Owner check at any time: `python -m agentics.cli verify`
(accept reviewed changes with `--refresh`). Drift alerts; it never kills the
loop — the owner decides.

---

## 3. H2 — the owner always knows what is happening

Three channels, all backed by SQLite, all readable offline:

1. **Published intent** — every node calls `ctx.report(...)`, storing the
   agent's current node, status, task, and a structured intent
   (`goal / why / steps / expected_files`) on the heartbeat row
   (`heartbeats.current_node`, `.task_id`, `.intent`).
2. **Activity journal** — an append-only feed (`journal` table) of what/why
   each agent did: `claim`, `open_mr`, `review`, `merge`, `cycle_done`,
   `task_created`, `task_exempt_set`, `integrity_alert`, `startup`… The table
   is trimmed to the newest rows automatically, so a forever-running lab
   stays bounded.
3. **Surfaces**:
   - `python -m agentics.cli status` — live agents (node + intent), task
     counts, claims.
   - `python -m agentics.cli watch` — terminal live view (agents with
     intent, claims, merge queue, backlog), `--once` for a snapshot.
   - `python -m agentics.cli events -f` — tail the activity journal live in
     the terminal (`--agent a02` to filter, `-n` for history); without `-f`
     it prints the recent story oldest-first.
   - `python -m agentics.cli dashboard` — web UI at `http://127.0.0.1:8080`:
     per-agent intent cards, task board, MR pipeline, merge queue, memory
     health, live journal (polls `/api/state` every 2.5 s, zero external
     assets, works fully offline).
   - `python -m agentics.cli metrics` — how the lab itself is evolving
     (merge rate by category, from the traces).

---

## 4. Running it

```bash
pip install -r requirements.txt          # no GITHUB_TOKEN needed anymore

python -m agentics.cli run               # start the swarm (local-only mode)
python -m agentics.cli dashboard         # owner's window (another terminal)
python -m agentics.cli task "Improve the benchmark runner" \
        --category code --body "what & why"    # feed work as the owner
python -m agentics.cli status            # CLI snapshot
python -m agentics.cli verify            # runtime-tree integrity check
python -m agentics.cli stop              # graceful stop (STOP file)
python -m agentics.cli reset-memory --dry-run   # preview a topic reset
```

Merge behaviour: with `AUTO_MERGE` on (default) the Integrator merges
approved MRs serially after tier-2 passes; with `AUTO_MERGE` off the queue
only records readiness and the owner merges by hand.

### 4b. Switching research topic — `agentics reset-memory`

Every memory surface is topic-bound, so a new topic needs a clean slate —
old findings would be recalled into the new research, old lessons would
steer new plans, old workspaces and origin branches would shadow fresh
clones. `agentics reset-memory` (added after `repair`) wipes exactly that
set and nothing else:

- **wiped** — `data/shared_memory.db` (+`-wal`/`-shm`: tasks, findings,
  panels, journal, cached role prompts), `data/chroma` (vector recall),
  `data/personal/` (per-agent lessons), `data/workspaces/` (old checkouts),
  `data/lab.git` (old-topic branches/MRs — re-seeded from the checkout at
  the next boot by `LocalRepo.ensure`);
- **kept** — lab code, the integrity baseline (`data/integrity`),
  `accounts/`, `configs/`, `.env`, `logs/`.

Flags: `--dry-run` (list what would go, delete nothing), `--yes` (skip the
confirm prompt), `--keep-repo` (preserve `data/lab.git`), `--force`
(override the running-agent guard). The guard refuses to run while any
agent heartbeat is fresher than 120 s and not marked `stopped` — a reset
under a live swarm would corrupt sessions; stop the lab first
(`agentics stop`). After the reset, change `research_topic`
(`configs/lab.yaml` wins over `RESEARCH_TOPIC` in `.env`) and start the lab
again — schema, bare origin and vector store are rebuilt empty.

### 4c. Grading the swarm itself — `agentics swarm-eval`

Output quality alone cannot judge a multi-agent lab: one lucky answer can
come from premature convergence or a single agent carrying the rest
(swarm-intelligence evaluation literature calls for trajectory,
collaboration and hierarchy metrics, not just per-answer scores).
`agentics swarm-eval` (`agentics/swarm_eval.py`) grades the lab AS A SWARM
from its own immutable traces — no LLM, read-only — in five groups:

1. **Coordination & task** — merge rate (judged traces), share of decided
   panels that had >= 2 distinct reviewers (cross-agent verification),
   citation-domain coverage of findings, share of active agents that ever
   produced a terminal outcome;
2. **Convergence & dynamics** — panel approval rate (converged to correct),
   mean vote rounds and mean cycle to a verdict, defense save rate;
3. **Trajectory & handoff** — productive-vs-looping cycle ratio
   (`cycle_done` outcomes vs `no_change`), node events per cycle
   (trajectory length), claim churn and lock friction, cycle-error share;
4. **Resilience & scale** — LLM ok-rate, budget-breaker trips, error
   recovery (a `cycle error` followed by that agent's next `cycle_done`),
   plan-retry/rework recovery, replication counts, and the scalability
   elasticity **S = %ΔP / %ΔN** across persisted snapshots at different
   swarm sizes (S > 1 → superlinear);
5. **Governance & cost** — quality gap (distinct objectors on rejected vs
   approved panels), peer-substantiated incoherence rate (objection stood
   after defense / author conceded), LLM calls + wall-clock seconds per
   merged MR (cookie providers expose no token counts — the ledger's
   honest units).

Flags: `--days N` (window), `--json` (full machine-readable report),
`--no-save` (don't persist). Every real evaluation appends one row to
`swarm_eval_snapshots` — the history that makes S and trend reading
possible. Missing data renders as an honest `n/a`, never a fabricated
number. Tests: `tests/test_swarm_eval.py`.

---

## 5. Testing

`python -m pytest -q` — 214 tests, no network required:

- `tests/test_local_ops.py` — task CRUD, claim lifecycle, backlog seeding,
  merge-queue semantics (serial, conflict retries, AUTO_MERGE off).
- `tests/test_local_review_merge.py` — full claim → implement → review →
  merge lifecycle on a **real** git bare repo, conflict path, author blocked
  from pushing agent code.
- `tests/test_protected_guard.py` — P10 semantics (normalisation, task
  exemption, env override, credentials never exemptable).
- `tests/test_integrity.py` — manifest build/verify/drift, orchestrator
  startup baseline (layer 4).
- `tests/test_dashboard_api.py` — dashboard state endpoint.
- `tests/test_scan_cache_stability.py` — legacy `github_ops` scan cache keyed
  on stable identity only (CPython `id()` reuse could resurrect a stale list).
- `tests/test_journal_cap.py` — journal trim keeps the feed bounded.
- `tests/test_reset_memory.py` — topic reset: dry-run deletes nothing,
  running-agent guard, complete wipe (DB/WAL/chroma/personal/workspaces/
  origin), `--keep-repo`, cold lab, CLI paths.
- `tests/test_swarm_eval.py` — the five swarm-metric groups over a seeded
  history (panels with defense, findings, LLM ledger, error recovery),
  scalability elasticity S across snapshots, honest `None` on empty data,
  CLI rendering + snapshot persistence.

---

## 6. Known limitations

- **Single machine** — shared memory is one SQLite file (WAL) and "origin" is
  one bare repo; scale-out needs an actual server or a sync layer.
- **Merge channel** — merges are fast-forward pushes to `data/lab.git`; if
  two Integrators ran concurrently the second push would fail as
  non-fast-forward (by design there is exactly one Integrator, P4).

---

## 7. Reconciliation log (multi-agent integration decisions)

This branch is built by several agents pushing to ONE branch; the later
pusher must cross-check the earlier work and only integrate what genuinely
complements it. Log of the reconciliations that produced this file:

**A. `main` merged in (408e7a5 — production reliability).** The branch was
cut before main gained: LLM failures degrade to `state["error"]` → reflect
(claim released, trace + lesson recorded) instead of killing the LangGraph
task; Hyperbrowser fail-fast challenge detection; GH pacing/retry/scan-cache
for the dormant github_ops (still unit-tested). Orchestrator conflict
resolution kept this branch's `is_auth_failure` classifier (more precise than
the heuristic main removed) — the policies compose: graph nodes no longer
raise, so only genuine auth failures cool a cookie. The prod-fix review test
was ported from the GitHub-PR fixture to the local MR flow.

**B. Pre-receive hook — parallel proposals deduplicated.** Two agents built
the git layer independently. The shipped `origin_hook_script` was kept (owner
identity by committer email, per-commit checks, main protection); the
parallel version was dropped. One delta was kept from it: `ensure_guard` now
generates the hook from the RUNTIME `protected_paths` (lab.yaml /
`PROTECTED_PATHS`), so extending the list at the app layer also extends the
git layer — one source of truth, no drift.

**C. Backlog rotation — hybrid.** The original top-up always re-opened
`SURVEY_ANGLES[0]`. Two parallel fixes landed: a least-used-first rotation
and a total-count rotation with walk-forward. Kept: least-used ordering
(robust against owner tasks inflating a counter) + walk-forward when a title
is still open elsewhere, PLUS a new first-priority source neither had: the
newest un-investigated failed trace becomes a bugfix task (GEA loop; an
investigated failure never re-opens).

**D. Terminal live view — one command, not two.** `events` already read the
journal; it gained `-f/--follow` (live streaming) instead of adding a
duplicate `journal` command. `watch` remains the full-screen live view.

**E. Deliberate non-ports.** The parallel GitHub-API emulator (duck-typed
PyGithub) proposal: dropped — rewriting the nodes onto the SQLite task board
(cd8b536) is cleaner and has no emulation layer to keep in sync. A separate
`events.jsonl` event bus: dropped — the `journal` table is the single source
of truth shared by the dashboard, `events`, `watch` and `explain`. A
standalone smoke script: dropped — `tests/test_local_review_merge.py` already
is a real-git E2E that doubles as the offline health check.

---

## 8. Adaptive planning & role prompts (LLM-authored, `agentics/planning.py`)

Two layers used to sit hardcoded between the swarm and the model: the static
role missions (`ROLE_MISSIONS` + a fixed template) and the planning prompt,
whose JSON schema the model merely filled in. Both are now authored by the
LLM itself, with code owning only the safety rails.

### Role prompts — written by the model, reworked by evidence

- One ACTIVE mission per `(role, category)` lives in the shared `role_prompts`
  table — every agent of that role/category reads the same text, so the swarm
  stays coherent (one synthesis LLM call per TTL window, not per agent).
- The mission is written by a "prompt architect" meta-prompt (`SYNTHESIS_META`)
  that carries the role, the task category, the research domain, a digest of
  the role's recent traces (`role_experience`: what was tried, what failed)
  and, on a rework, the measured record of the previous mission ("has run N
  cycles: S ended well, F failed").
- Feedback loop: `node_reflect` feeds every finished cycle back into the
  active mission. `error` counts a failure, `yield` is neutral (a lost race
  is not the mission's fault), success decays the failure counter. After
  `role_prompt_rework_failures` (default 2) failed cycles the mission is
  superseded and the next cycle synthesizes a fresh one conditioned on what
  went wrong. TTL (`role_prompt_ttl_hours`, default 24) refreshes missions
  even without failures.
- **Guardrails are code-owned** (`roles.GUARDRAILS`): appended to EVERY
  system prompt after synthesis — never merged, never self-merge, protected
  paths, no secrets. The model may rewrite its mission; it can never relax
  the rules. `sanitize_mission` strips control characters and caps the
  length before the text ever reaches a prompt.
- Degradation: no LLM account, a failed synthesis or a refusal-length answer
  falls back to the static `build_system_prompt` behavior — the adaptive lab
  turns back into the pre-adaptive one, never into a crash.
- Config: `adaptive_prompts: true`, `role_prompt_ttl_hours`,
  `role_prompt_max_chars`, `role_prompt_rework_failures` (configs/lab.yaml;
  env `ADAPTIVE_PROMPTS`, `ROLE_PROMPT_TTL_HOURS`, …).

### Plans — proposed by the model, normalized by code

- The plan prompt (`PLAN_INSTRUCTION`) asks the model to PROPOSE: field one
  is `approach` — what will be done and why, alternatives considered and
  rejected, risks, what counts as success. This is what the owner reads
  first in the intent report.
- The machine contract (`PLAN_CONTRACT`) is the only fixed part: `files`
  (P13 scope locks), `steps` (intent), `commit_message`, `verify_cmds` and
  `hypotheses` (P16 empirical verification), `reuses` (tool registry) — the
  fields the lab consumes mechanically.
- `normalize_plan` is a tolerant adapter: it extracts the contract with
  defensive caps (runaway answers, not structure) and preserves everything
  else the model adds — `state["plan_approach"]`, `state["plan_extras"]`
  (JSON of the model's own fields, shown to the owner) — instead of dropping
  them by a rigid schema. A plan without `verify_cmds`/`hypotheses` behaves
  exactly like the pre-adaptive plans (backward compatible).

### Plan-conditioned downstream prompts (no fixed recipe left)

- The per-file implement ask (`node_implement`, `_implement_brief`) is no
  longer a minimal fixed template: it carries the planner's OWN proposal —
  the declared `approach`, the P16 `verify_cmds` that will judge the code
  and the pre-registered hypotheses — so the implementer writes code that
  satisfies the checks it itself declared. Bounded (approach 700 chars,
  5 cmds, 2 hypotheses); a minimal/legacy plan reproduces the old prompt
  byte for byte. Zero extra LLM calls.
- The research query (`node_research`) now carries the issue's own problem
  statement (a bounded 400-char body excerpt) instead of only the title +
  a fixed recipe. Dedup still keys on the title-based topic, so the P7
  skip guard is untouched.
- Both paths are covered offline in `tests/test_plan_conditioned_prompts.py`
  (brief content, runaway caps, legacy-plan backward compat, body excerpt
  bound).
