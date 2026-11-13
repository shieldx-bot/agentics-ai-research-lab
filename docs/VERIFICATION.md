# VERIFICATION — agents evaluate ideas by RUNNING code, not by asking an LLM

> Answers the owner's three hypotheses directly:
>
> **H1 — Can agents use cmd / code / tests / review like a real researcher?**
> Yes — every check is a real process execution through the guarded workspace
> runner (allowlisted interpreter, scrubbed environment, bounded lifetime):
> declared `verify_cmds`, pytest suites, and self-contained experiment
> protocols. Cross-review then weighs the recorded evidence, not prose.
>
> **H2 — Can an agent verify its own hypothesis?**
> Yes — a claim is pre-registered with a QUANTITATIVE criterion
> (`metric / op / target`) BEFORE any code runs; the protocol prints
> `AGENTICS_RESULT: {"metric": <number>}`; the verdict is a machine
> comparison of measured vs criterion — never an LLM's self-assessment.
> Protocols run twice; non-reproducible or unparsable output is
> `inconclusive` and can never pose as success. And the verdict does not
> rest on one agent's word: ANOTHER agent independently re-runs the stored
> protocol in its own workspace (P24/P24bis replication) before the claim
> is marked `replicated`.
>
> **H3 — Can an agent share its verification results with every AI?**
> Yes — verdicts land in the shared `evidence` + `hypotheses` tables,
> auto-broadcast to the swarm (messages + journal), rendered on the owner's
> dashboard, queryable via `agentics evidence` / `agentics hypotheses`, and
> injected into EVERY agent's prompt as settled knowledge (supported facts
> become reusable common ground; refuted claims stop being repeated).
> Sharing is also enforced: the replication ledger tracks WHO re-ran WHAT,
> so the swarm never pays for the same experiment twice — and a refuted
> replication self-corrects the shared knowledge everywhere.

The stack grew from several parallel contributors on this branch; this
document describes how the pieces compose today.

---

## 1. The pieces and who runs them

| layer | module | who triggers it | verdict source |
| --- | --- | --- | --- |
| declared checks (MR flow) | `verify_ops.py` (`run_experiments`) | Coder/Writer: `verify_cmds` declared in `node_plan`, executed in `node_implement` before push | process exit code (`confirmed` / `refuted` / `inconclusive`) |
| hypothesis contract (MR flow) | `agentics/verify/` (`hypothesis.py`, `experiment.py`: register_hypothesis → run_hypothesis → settle_hypothesis) | Coder/Writer: `hypotheses` declared in `node_plan` (claim + prediction + criterion + experiment code) | measurement vs criterion, re-adjudicated from the protocol's `AGENTICS_RESULT` output, 2-run reproducibility gate |
| plan-hypothesis bridge (MR flow) | `graph/nodes.node_implement` composing the `agentics/verify/` glue (register → run → settle per hypothesis) | same declared `hypotheses` — the claim is pre-registered BEFORE the run, the protocol executes twice, and the settled verdict + protocol file join the MR record | marker path (`AGENTICS_RESULT`); missing / unparsable / non-reproducible output is `inconclusive`, never guessed through |
| research claims (researcher flow) | `verify_metrics.py` (`Criterion`, `run_protocol`) + `graph/nodes.node_verify` | Researcher: after storing a finding, `node_verify` extracts up to 2 quantitative claims, pre-registers each with a criterion, writes one protocol per claim into `experiments/verify_h<id>.py` and runs it TWICE | measured metric vs pre-registered criterion + reproducibility check |
| shared ledger | `SharedMemory.record_evidence` / `attach_protocol_outcome` / `add_hypothesis` / `update_hypothesis` | every layer above | — |
| shared knowledge | `verified_knowledge()` → `_memory_context` → every system prompt; dashboard panel; `agentics evidence` / `agentics hypotheses` CLI | automatic | — |

## 2. Vocabulary (one rulebook, two levels)

* **Evidence row verdict** (the check itself): `confirmed` | `refuted` |
  `inconclusive`. Exit 0 or a satisfied criterion is `confirmed`; a violated
  criterion is `refuted`; timeout (124), guard-blocked (126), unparsable or
  non-reproducible output is `inconclusive`.
* **Hypothesis status** (the claim's lifecycle): `proposed → testing →
  supported | refuted | inconclusive`. `confirmed` evidence moves the claim
  to `supported`.

A blocked or timed-out check is **inconclusive, never confirmed** — an agent
cannot "believe" its way past a guard.

## 3. The researcher flow, end to end

```
research (finding stored)
  → verify: LLM extracts quantitative claims + one self-contained protocol each
      → for each claim:
          pre-register hypothesis (criterion locked BEFORE the run)
          write experiments/verify_h<id>.py into the workspace
          run protocol 2x through workspace.run_command (guarded)
          parse the last AGENTICS_RESULT line (parse_metrics)
          adjudicate measured vs criterion (adjudicate — machine verdict)
          record_evidence (auto-broadcast) + attach metrics/runs
          update hypothesis status + journal 'hypothesis_verified'
  → reflect: verdicts are now shared knowledge
```

Protocol contract (enforced, violations become `inconclusive`): one
self-contained python file, standard library only, no network, deterministic
output, and exactly one final line `AGENTICS_RESULT: {"<metric>": <number>}`
measuring the metric the criterion names.

## 4. Why the criterion lives outside the protocol

An assert-based script CAN encode ">= 0.9" — but then the claim hides inside
generated code and the reviewer must trust it. Keeping the criterion as data
(`hypotheses.criterion`) next to the measured numbers (`evidence.metrics`)
makes every verdict auditable and replayable: the machine decision is
`measured <op> target`, and both operands are stored rows.

## 5. Reproducibility

A real researcher reruns the experiment. `run_protocol` executes each
protocol twice; if the two runs' metrics differ, the evidence is
`inconclusive` ("runs disagree — protocol is not reproducible") and the
hypothesis can never become `supported` on that evidence.

## 6. The reviewer runs code before it opines

`node_review` spends zero LLM tokens on an MR that cannot survive its own
measurements — mechanical gates run BEFORE any model opinion:

* **declared-checks re-run** — the reviewer re-executes up to 3 of the
  author's recorded `run_command` checks on the pushed branch in its own
  workspace (`cmds_from_evidence` → `run_checks_on_branch`, throw-away
  `review/tmp` branch, main always restored) and records them as
  `method="reviewer_rerun"` evidence; a re-run that refutes the author's
  check vetoes the approval outright ("evidence overrides model opinion");
* **protocol reproduction (P17)** — an exit-code re-run cannot judge a
  quantitative claim: a protocol always exits 0, the frozen criterion is
  the verdict. For the author's CONFIRMED protocol rows the reviewer
  re-runs the FULL two-run adjudication against the ledger's criterion
  (`run_protocols_on_branch`): a script that still prints AGENTICS_RESULT
  but misses the criterion is REFUTED here — invisible to any exit-code
  re-run;
* **the ledger follows the machine** — a refuted re-run (either path)
  re-verdicts the author's ORIGINAL evidence row (`set_evidence_verdict`,
  `verdict_by` trail), downgrades the hypothesis to `refuted` (so
  `verified_knowledge` stops presenting a dead claim as common ground)
  and journals `reviewer_rerun_refuted`. An inconclusive re-run (sandbox
  hiccup) only annotates the MR comment — never a false refutation. On
  the green path the MR comment carries the re-run digest, so the owner
  sees WHICH checks were reproduced;
* **tier-2 full-suite gate** (`verify_merge`) — the reviewer merges
  main+MR into a throw-away copy and runs the full pytest suite. This
  catches an MR that declared NO checks at all and mirrors the owner's
  `agentics mr merge` path and the Integrator's drain gate; failing it is
  `request_changes` with the reason, approvals carry
  `[mechanical: tier-2 passed]`. When the context has no tier-2 capability
  (node-level tools/tests), the prompt says so explicitly instead of
  pretending the tests ran.

## 7. Streaming — the swarm is never a blackbox

`run_cycle` drives the graph with `astream` and journals one digest line per
completed node (`event="node"`, with verdict/mechanical/review fields) —
`agentics watch` and the dashboard render each step live instead of waiting
for the cycle to end.

## 8. Owner surfaces

```bash
python -m agentics.cli evidence            # every recorded run (P16)
python -m agentics.cli hypotheses          # pre-registered claims + machine verdicts (P16bis)
python -m agentics.cli dashboard           # live panels: evidence + hypotheses
```

On the dashboard, the "Kiểm chứng thực nghiệm" card shows raw runs; the
"Giả thuyết đã đăng ký" card shows each claim's criterion, the measured
value and the settled status — pre-registration means the goalposts cannot
move after seeing the numbers.

## 9. Claim dedup — settled claims are reused, never re-run (H3 enforced)

Sharing verdicts is only half of H3; the other half is that the swarm never
pays for the same experiment twice. Every hypothesis row stores a normalized
`claim_hash` (case/whitespace-insensitive, `verify.hypothesis.claim_hash`).
Both verify paths consult `SharedMemory.settled_hypothesis_for_claim()`
BEFORE running anything:

* **research path** (`node_verify`): a claim already settled (supported /
  refuted) by any agent produces a `reused` result that references the
  first verdict — no new hypothesis row, no protocol run, no duplicate
  evidence row; the reuse is journalled (`hypothesis_reused`) for the owner.
* **coder path** (`node_implement`): a plan that re-declares an already
  supported claim proceeds to push at zero experimental cost; re-declaring
  an already REFUTED claim yields the cycle exactly like a fresh refutation —
  re-asserting a falsified claim cannot sneak past the gate by rerunning it.

Only `supported`/`refuted` count as settled: `inconclusive` claims always
deserve a fresh run. Pinned by `tests/test_hypothesis_dedup.py`.

## 10. Testing


`tests/test_verify_ops.py` (exit-code evidence), `tests/test_verify_metrics.py`
(criterion adjudication + reproducibility), `tests/test_evidence_memory.py`
(shared ledger), `tests/test_hypothesis_memory.py` (lifecycle + verified
knowledge), `tests/test_node_verify.py` (researcher flow node-level),
`tests/test_p16_bridge.py` (plan-declared hypotheses),
`tests/test_experiment_runner.py` (runner verdict paths, tier-2 gate,
streaming), `tests/test_reviewer_rerun.py` + `tests/test_p17_protocol_rerun.py`
(reviewer re-runs + protocol reproduction, incl. a real RepoWorkspace test
proving the protocol is checked out of the MR branch), plus the demo:
`python scripts/local_mode_demo.py` prints a `supported` and a `refuted`
hypothesis from real protocol executions.

## 11. P19 peer panel — the defense round is evidence-gated too

The panel (see POLICIES.md P19) is where verification meets veto: a single
reject vote opens a defense round, and the author's defense is only as good
as the runs behind it.

* **Vote with evidence** — `node_review` keeps the P16 re-run + P17
  reproduction machinery ahead of the vote: a refuted re-run votes `reject`
  even when the LLM says approve (evidence overrides model opinion), and the
  mechanical tier-2 gate votes `reject` without spending one LLM token.
* **Defend with runs** — `node_defend` executes the author's declared
  `extra_checks` (python/pytest only, allowlisted, bounded) in the author's
  own workspace and records them as evidence rows with
  `method="defense_check"` — the panel sees measurements next to the
  rebuttals; an inconclusive or refuted defense check is common knowledge
  the objectors weigh in round 2.
* **Finality is journalled** — every decisive outcome (`panel_approved`,
  `panel_rejected`, defense broadcasts, deadline sweeps) lands in the
  shared journal and the dashboard alerts card, so the owner sees WHO
  rejected WHAT and WHY, and the swarm learns from every dead MR (GEA).

Pinned by `tests/test_peer_review.py`, `tests/test_peer_review_memory.py`,
the defense paths in `tests/test_nodes_integration.py` /
`tests/test_layer4_guard.py` / `tests/test_experiment_runner.py`, and the
owner surfaces in `tests/test_dashboard_api.py`.

## 12. P23 replay — replication runs inside the live cycle (H2 + H3 closed)

A settled verdict is a measurement, and measurements go stale: dependencies
move, protocols drift, an early run may simply have been wrong. The replay
engine (`agentics/verify/replay.py`) re-runs a SETTLED claim from its stored
protocol source (registered with `protocol_code` since the P23 precondition
commit) and re-settles it from fresh measurements:

| outcome | event | effect |
| --- | --- | --- |
| same verdict | `claim_reconfirmed` | the claim survives scrutiny, `attempts` grows |
| supported → refuted | `claim_falsified` | verified_knowledge corrected automatically |
| refuted → supported | `claim_overturned` | the original refutation was wrong |
| settled → inconclusive | `claim_unstable` | the protocol no longer runs deterministically |

Every replay lands in the SAME shared evidence table (`method="replay"`)
with the measured metrics attached, so the whole swarm sees replication
results — H3 at its strongest: verdicts are shared WITH their reproducible
artifacts, and a second agent can re-check the first agent's word.

**Live wiring (this commit):**

* the supervisor's `node_scan` runs a bounded sweep on schedule —
  `REPLAY_SWEEP_SIZE` claims (default 2, `0` disables) every
  `REPLAY_EVERY_N_CYCLES` cycles (default 3) — following the exact
  P13/P19/P22 maintenance-sweep pattern: replication must never break the
  scan, and the scan summary line carries the drift events to the dashboard;
* the owner can run the sweep on demand: `agentics replay [--limit N]`
  or a single claim with `agentics replay --id X`;
* contradictions between same-criterion hypotheses stay owner-visible via
  the `claim_contradiction` journal alert (P23 memory layer).

Pinned by `tests/test_replay.py` (engine), `tests/test_protocol_code.py`
(storage precondition) and `tests/test_replay_wiring.py` (live-cycle wiring:
on-schedule sweep, off-schedule skip, zero-off, drift falsification
correcting verified_knowledge, and a broken protocol never breaking the scan).

## 8. Re-verification (P23): settled claims must survive being re-run

A ledger nobody re-checks is a diary. The replay layer lets the owner (or a
supervisor sweep) re-run SETTLED claims from the protocol source stored with
the hypothesis (`hypotheses.protocol_code`), in a fresh `owner-replay`
workspace — so replication survives P21 workspace pruning:

```bash
python -m agentics.cli verify-all            # replicate newest settled claims
```

Each replay records fresh evidence (method='replay'), bumps the claim's
`attempts`, and re-settles the hypothesis:

| replay outcome | journal event | effect |
| --- | --- | --- |
| same verdict | `claim_reconfirmed` | attempts++ — the claim has survived N scrutinies |
| supported → refuted | `claim_falsified` | drift: `verified_knowledge` is corrected automatically |
| refuted → supported | `claim_overturned` | the original refutation was wrong (or the code improved) |
| now inconclusive | `claim_unstable` | protocol no longer runs deterministically — reviewer decides |

**Contradiction alerts:** when ANOTHER hypothesis with the SAME criterion
settles to the OPPOSITE verdict, `update_hypothesis` journals
`claim_contradiction` — the swarm disagreed with itself on a measurable fact
and the owner/panel arbitrates. A replay flipping its own row is drift, not
disagreement, and does not alert.

Tests: `tests/test_protocol_code.py`, `tests/test_replay.py` (including a
real bare-origin/clone integration run — the exact `verify-all` sequence).

## 13. P24/P24bis — INDEPENDENT replication by another agent (H2 closed twice)

The P23 sweep re-checks claims over TIME from one workspace — a claim's
verdict still rests on a single agent's word on day one. Independent
replication is the other half: a DIFFERENT agent re-runs the claim in its
OWN workspace, and only then does `verified_knowledge` flag it `replicated`.

Two pieces, one rulebook:

* **P24 ledger** (`SharedMemory`): `hypotheses.replication` +
  `replication_attempts` — the canonical once-per-swarm bookkeeping.
  `hypotheses_needing_replication(agent)` offers each agent the OLDEST
  settled `supported` claim it did NOT author (author exclusion, retry
  budget: an `inconclusive` replication is retried by a DIFFERENT peer up
  to 2 attempts, then left alone).
* **P24bis engine** (`agentics/verify/replication.py`):
  `replicate_candidate` rebuilds the stored protocol in the peer's
  workspace, runs the full two-run reproducibility gate, writes the
  provenance ledger (`record_replication_detail`: who, measured metrics,
  per-run detail) and the peer's own evidence row
  (`method='replication'`), then applies the P17 asymmetry:

| replication outcome | journal event | effect |
| --- | --- | --- |
| confirms | `replication_confirmed` | claim annotated `replicated` — author-confirmed becomes independently replicated |
| refutes | `replication_refuted` | ledger closed: the author's evidence row is re-verdicted, the claim downgraded — `verified_knowledge` stops serving a dead claim |
| inconclusive | `replication_inconclusive` | annotates only, NEVER refutes; another peer may retry within the budget |

The live wiring is `node_verify` (`agentics/graph/nodes.py`): BEFORE any
fresh-finding verification — and needing neither a finding nor an LLM —
every researcher cycle feeds the P24 selector into the engine
(`_replicate_settled`, up to 2 claims per cycle). A researcher with nothing
new to verify still advances the swarm's trust in what it already knows.
Self-replication is refused at BOTH layers (selector + engine).

Composition with dedup (section 9): an agent that meets an already-settled
claim replicates it once AND reuses the settled verdict — efficiency and
trust do not compete.

Consumption + routing (P24bis surfaces): `verified_knowledge()` carries a
`replicated` flag — every agent's prompt annotates independently replicated
claims as `SUPPORTED+ (independently replicated)`, the strongest kind of
common ground; the dashboard has a Vietnamese "Lặp lại độc lập" panel
(counts from the canonical ledger + provenance rows). Reviewers also carry
a dedicated route (`node_replicate`, knob `REPLICATION_LIMIT`, default 1
per reviewer cycle, 0 = off): pending replication outranks ordinary review
(but never the P19 defense), so the trust work happens even when no new
finding needs verifying.

Tests: `tests/test_replication_ledger.py` (selection + bookkeeping),
`tests/test_replication_memory.py`, `tests/test_replication_engine.py`,
`tests/test_replication.py` (live node wiring, zero-LLM path),
`tests/test_node_replicate.py` (reviewer route + routing table),
`tests/test_replication_surfaces.py` (prompt + dashboard), plus the
dedup composition pinned in `tests/test_hypothesis_dedup.py`.
