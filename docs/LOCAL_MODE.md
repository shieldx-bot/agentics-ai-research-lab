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

## 2. H1 — the agent code cannot be corrupted (4 layers)

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
`configs/secrets.yaml`) are **never** exemptable.

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
```

Merge behaviour: with `AUTO_MERGE` on (default) the Integrator merges
approved MRs serially after tier-2 passes; with `AUTO_MERGE` off the queue
only records readiness and the owner merges by hand.

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

---

## 6. Known limitations

- **Rework loop** — an MR marked `changes_requested` currently waits for a
  human nudge: the author's claim was handed off at MR time, so nothing
  automatically returns the task to its author for a fix-up push. Suggested
  next step: let the author re-claim a task whose latest MR got
  `changes_requested`, push fix commits to the same branch, and re-open the
  review.
- **Single machine** — shared memory is one SQLite file (WAL) and "origin" is
  one bare repo; scale-out needs an actual server or a sync layer.
- **Merge channel** — merges are fast-forward pushes to `data/lab.git`; if
  two Integrators ran concurrently the second push would fail as
  non-fast-forward (by design there is exactly one Integrator, P4).
