# QD ideation closed loop (IDEAgent gate + Kosmos harvest + autoresearch ledger)

Deepens commit `4011788` (world model + diversity gates): the loop no longer
only *filters* ideas — it *generates* them from experiment results, *judges*
them before they cost a protocol run, and *remembers* every attempt.

## The loop, end to end

```
registry backlog (proposed + inconclusive retries)
        |
        v
deep-dive: LLM writes protocol -> machine verdict (P16 gate)
        |
        v
settle -> experiment LEDGER row (autoresearch results.tsv)
        -> world model note (Kosmos: verified/refuted/gap)
        -> INSIGHT HARVEST: 0-2 follow-up children (Kosmos closed loop)
                each child: PEM near-dup gate -> IDEAgent judge ->
                accept | one repair | reject (-> rejected-pattern archive)
        |
        v
topic saturated + registry EMPTY
        -> GAP PROPOSAL: the world model proposes ONE gap-closing
           hypothesis (same gate pipeline) — the lab pivots, never idles
```

## Mechanisms by source

### Kosmos (arXiv 2511.02824) — close the loop
- **Insight harvest** (`_harvest_insights`): after a deep-dive settle
  (supported OR refuted — refutations still open questions), one ideation
  call proposes up to `IDEA_MAX_CHILDREN` follow-ups grounded in the
  measured result. Children carry `parents=[settled_hyp_id]` (lineage) and
  the IDEAgent quintuple signature.
- **World model as task proposer** (`_propose_gap_hypothesis`): when the
  literature topic is saturated AND the registry is empty, the agent reads
  `world_notes` (gaps/questions) and registers one gap-closing hypothesis
  instead of skipping. The task closes only when even the pivot fails.

### IDEAgent (arXiv 2607.22375) — qualify before you burn a run
- **Judge** (`JUDGE_PROMPT`): non-obviousness / clarity / feasibility
  (0-100) + soundness (0-9).
- **Qualification gate** (`qualification()`): `Q = 0.7*N + 0.2*S + 0.1*C`;
  accept at `tau >= IDEA_MIN_SCORE` (60); near-misses within
  `IDEA_REPAIR_MARGIN` (20) of every threshold get ONE repair round
  (critic feedback -> fixed child, re-judged, must fully pass); anything
  weaker is rejected outright.
- **Rejected-pattern archive** (`rejected_patterns` table): compact
  mechanism + one-line reason, capped at 100 rows, rendered into every
  ideation seed prompt — the loop pivots away from failing mechanisms
  (full rejected ideas would bloat context).
- **Archive refresh per candidate**: a child registered moments ago
  already blocks its own clones (no self-cloning inside one harvest).

### autoresearch (karpathy/autoresearch) — loop discipline
- **Experiment ledger** (`experiment_ledger` table, max 500 rows): every
  settle — deep-dive AND coder path (`settle_hypothesis`) — lands with
  keep/discard semantics: supported=keep, everything else=discard, plus
  metric/measured/op/target so runs stay comparable.
- **Near-miss retries**: `inconclusive` hypotheses (protocol measured
  nothing usable) stay testable until `IDEA_MAX_ATTEMPTS` (2); the retry
  prompt carries the previous failure note ("fix it if trivial, change the
  approach if broken, keep it simple"). Retries run BEFORE fresh claims.
- **Retire after failed retry**: a retry that yields no parsable protocol
  retires the claim as refuted (`hypothesis_retired` journal event) — the
  idea, not the protocol, is fundamentally broken.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `AGENTICS_IDEATION` | `1` | `0` disables the whole ideation loop (no ideation LLM calls) |
| `IDEA_MIN_SCORE` | `60` | IDEAgent qualification gate tau (0-100) |
| `IDEA_REPAIR_MARGIN` | `20` | near-miss band for the single repair round |
| `IDEA_MAX_CHILDREN` | `2` | follow-ups one settled experiment may spawn |
| `IDEA_MAX_ATTEMPTS` | `2` | deep-dive retries of inconclusive protocols |

## Cost bounds

One deep-dive settle costs at most: 1 ideation call + `IDEA_MAX_CHILDREN`
judge calls + (repair band) 2 extra calls per repaired child. All calls run
with `purpose="ideation"` (routed to the general pool), pass the P20 budget
gate, and degrade to "no ideation" on any failure — the experiment itself
never depends on the ideation round. The mechanical gates (PEM near-dup,
coverage CCR) run before any LLM call and cost nothing.
