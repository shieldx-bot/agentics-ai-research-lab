# Research-loop diversity & world model (DAPS / IDEAgent / Kosmos)

Owner-facing summary of the 2026-11-14 upgrade driven by three papers:

| Paper | Mechanism ported | Where it lives |
|---|---|---|
| **Kosmos** (arXiv 2511.02824) — AI scientist with a structured world model | `world_notes` table: structured shared state (findings / verified / refuted / gaps) appended every cycle, rendered into every agent snapshot, the research query and the plan prompt | `SharedMemory.world_add/world_render`, `_world_model_text()` |
| **IDEAgent** (arXiv 2607.22375) — ideation as Quality-Diversity search | Hypothesis **signature quintuple** `{problem, mechanism, novelty, assumptions, effect}` stored per hypothesis; diversity judged on the mechanism axes, not wording | `hypotheses.signature`, `diversity.signature_text()` |
| **"Beneath the Diff" / DAPS** (arXiv 2609.00077) — algorithmic mode collapse in autonomous research loops | **PEM** near-duplicate gate (reject re-worded clones), **CCR** coverage reweighting (rare mechanism categories first), **entropy audit** (mode-collapse alarm) | `_pem_gate()`, `qd.coverage_order()`, `node_scan` `diversity_audit` |

## Why (live symptoms, 2026-11-14)

- 810 traces collapsed into **4 clusters**, 0 skills distilled: the swarm kept
  producing variants of the same work — exactly DAPS's *algorithmic mode
  collapse*: surface diversity stays high (new wording, new files) while the
  MECHANISM repeats.
- Evidence ledger grew mostly from duplicate replays; hypothesis stream
  narrowed onto one pattern.

## What changed at runtime

1. **Every cycle (supervisor scan)**: `diversity_audit` journal event with the
   mechanism entropy (nats) over the last 20 hypotheses and the categories
   seen. When entropy < `DIVERSITY_MIN_ENTROPY` (default **1.0**) with ≥8
   samples, an owner alert fires naming the collapsed categories.
2. **Every new hypothesis (verify + deep-dive)**: the **PEM gate** compares
   the claim (light stemming + stopword Jaccard, plus the signature axes)
   against the last 12 hypotheses. Score ≥ `DIVERSITY_JACCARD_MAX` (default
   **0.8**) → rejected as a near-clone (`hypothesis_near_duplicate` journal).
   A registry proposal that re-words a SETTLED claim inherits that verdict
   instead of burning a protocol run.
3. **Deep-dive selection**: instead of oldest-first, the registry backlog is
   ordered by **category coverage** (rarest mechanism category in the recent
   stream first) — CCR — so the swarm is pushed toward under-explored
   directions.
4. **World model**: findings, settled facts and refutations are appended to
   `world_notes` with provenance (`finding#N`, `h#N`). The render is injected
   into agent snapshots (`world`), the research query and the plan prompt
   ("build on VERIFIED, close a GAP, never re-assert REFUTED"). Bounded to
   `WORLD_CONTEXT_NOTES` (default **12**) lines per prompt, 400 rows on disk.

## Environment knobs

```bash
DIVERSITY_JACCARD_MAX=0.8    # PEM threshold (lower = stricter novelty floor)
DIVERSITY_MIN_ENTROPY=1.0    # nats; below this the supervisor alerts
WORLD_CONTEXT_NOTES=12       # world-model lines carried into each prompt
```

The mechanism taxonomy is frozen in `agentics/diversity.py`
(`_MECHANISM_KEYWORDS`, 10 categories + `other`), mirroring DAPS's
pre-registration discipline: derive once, then freeze — change it only with a
reason logged in the journal.
