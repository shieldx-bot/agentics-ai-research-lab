# Advanced Stack — Wave 3 proposal (TOP 5)

Owner-facing proposal, 2026-11-14. Builds on wave-1 and wave-2
(`docs/ADVANCED_STACK.md` on `main`: PUCT frontier, MAP-Elites, local
router, GraphRAG-lite, Elo Arena / skill library, prompt evolution, EIG,
verify calibration, reflection tree) and the world-model/diversity stack
(`docs/DIVERSITY_WORLD_MODEL.md`). **None of the five proposals below
duplicates an existing technology** — each was checked against the tree
(`update_interactive_local` @ `7d9260a` + `main` @ `6f7c29f`).

Design rules inherited from waves 1–2:

* selection signal is **exit-code truth only** — never an LLM's opinion;
* all generation/mutation/extraction runs on the LOCAL router when
  available (zero cookie cost);
* every consumer **degrades to the previous behavior** when its flag is
  off or its dependency is absent;
* tables stay bounded by design; every feature journals a KPI event.

---

## Overview

| # | Proposal | One line | Effort | Default |
|---|----------|----------|--------|---------|
| 1 | Test-time compute scaling (best-of-N + machine gate) | sample N candidates, keep the one that PASSES | M | off (`BTC_N=1`) |
| 2 | Formal verification track (Z3 / SMT) | verdict level `proven` — prove invariants, not just measure | M–L | off (`FORMAL_VERIFY=1`) |
| 3 | Hermetic replay (environment fingerprints) | replication that survives dependency drift and machine moves | S | on (`REPLAY_ENV_FINGERPRINT=1`) |
| 4 | Federated knowledge exchange | labs exchange signed evidence bundles; foreign verdicts are re-run locally before acceptance | M | manual (CLI only) |
| 5 | Auto-ablation engine | every supported claim earns a causal mini-study (delta vs baseline) | M | off (`ABLATION_RATE=0.3`) |

---

## 1. Test-time compute scaling — best-of-N with machine-gate selection

**What.** The implement / repair / protocol-writing paths currently ask the
model ONCE and hope. Instead: sample N candidates (local router for
plumbing-grade purposes, cookie pool for science-grade ones), run each
candidate through the EXISTING machine gates — tier-1 (`py_compile` +
`pytest -x`) for code, `run_protocol` for protocols — and keep the first
candidate that passes. The gate becomes a selection engine, not just a
checkpoint.

**Why advanced.** "Best-of-N + verifier reranking" is the paradigm that
turned LLMs from text generators into problem solvers (test-time compute
scaling). The lab already owns the rare ingredient — cheap, trustworthy
machine verifiers — but spends it as a filter, not as a selector. This
converts the same exit-code truth into accuracy gains without any model
upgrade, and pairs perfectly with the P16-R self-repair loop: N candidates
replace one repair attempt.

**Integration.**
* new `agentics/llm/sampler.py`: `best_of_n(ctx, purpose, prompt, judge,
  n)` — sampling loop, per-candidate gate, ranked result
  (`winner_rank`, `pass_rate`).
* `graph/nodes.py` — `node_implement` (tier-1 judge), `verify/repair.py`
  (protocol judge); behind `BTC_N` (default 1 = today's behavior) and
  `BTC_PURPOSES` (default `implement,repair,protocol`).
* Sampling runs on the local router when the purpose allows it — N>1
  stays free; otherwise N cookie calls with P20 budget accounting.

**Verdict & selection signal.** unchanged vocabulary; the gate is
exit-code truth. Journal: `btc_selected {purpose, n, winner_rank,
pass_rate}`.

**KPI.** first-attempt tier-1 pass rate and repair-cycle count before/after
(`journal` events `btc_selected` vs `p16_repair_*`).

**Risks.** CPU cost scales with N on the local router — cap N<=4 and lean
on `LOCAL_LLM_NUM_PREDICT`; cookie-mode best-of-N must respect the P20
budget (`policies/budget.py` refuses the extra calls like any other ask).

## 2. Formal verification track — a `proven` verdict with Z3

**What.** A hypothesis criterion gains `kind`: `metric` (today) or
`invariant` (new). An invariant protocol declares a property that must
hold for ALL inputs in a bounded space; `verify/formal.py` compiles the
declared property to an SMT query (Z3, MIT license, pip-installable, runs
local) and records the evidence verdict **`proven`** — stronger than
`confirmed`, rendered as `[PROVEN]` in `verified_knowledge`.

**Why advanced.** Measurement generalizes from finite samples; SMT
disproves/proves over an infinite space. A lab whose evidence vocabulary
includes PROVEN (not just measured) reaches a fundamentally higher
epistemic bar, and PUCT Q / Elo priors can weight proven claims above
confirmed ones — the whole swarm inherits the stronger signal.

**Integration.**
* `verify_metrics.Criterion` gains `kind` + `invariant` fields
  (backward-compatible: missing kind = metric);
* new `agentics/verify/formal.py` — Z3 dispatch, counterexample
  generation on refutation (a counterexample IS a measurement the
  protocol can print: `AGENTICS_RESULT: {"counterexample": ...}`);
* `run_protocol` dispatches by kind; evidence verdict vocabulary and
  `VERIFY_STATUS` gain `proven`; verify prompts teach the contract.
* Writing good invariants is hard — that stays a SCIENCE purpose (cookie
  models); Z3 checking itself is local and free.

**KPI.** `journal` event `formal_proven`; counterexample yield on
invariant refutations; (long-run) share of PROVEN rows in
`verified_knowledge`.

**Risks.** LLM-written invariants can be trivially true — the contract
requires a mandatory sanity check (the protocol must demonstrate the
property on at least one concrete input, or the verdict stays
inconclusive). Z3 absence = feature off, zero behavior change.

## 3. Hermetic replay — environment fingerprints for replication

**What.** P23 replay currently re-runs protocols in whatever environment
the replayer happens to have. Each protocol gains a runtime fingerprint —
python version, declared package set (`requirements.lock` the protocol
may carry), interpreter path class — recorded with the evidence row.
Replays compare fingerprints:
* same env + same verdict → `claim_reconfirmed` (today's path);
* **different env + same verdict → `cross_env_confirmed`** — strictly
  stronger replication evidence;
* different env + different verdict → **`env_sensitive`**, not
  `claim_falsified` — the claim did not fail, its environment did.

**Why advanced.** Scientific replication without environment identity is
half-blind. This is the cheapest technology on this list and the one that
protects everything the lab already built: the ledger this lab accumulates
now must still be replayable in six months and on a different machine
(sandbox ↔ VPS ↔ home) — the swarm runs in three places already.

**Integration.** `verify/replay.py` (fingerprint record/compare),
`verify/replication.py` (same check), evidence `meta` carries the
fingerprint; knob `REPLAY_ENV_FINGERPRINT` (default 1), `env_sensitive`
joins the drift-event vocabulary consumed by verify calibration (wave-2
#4 must not count env-sensitive flips as calibration failures).

**KPI.** share of `cross_env_confirmed` among replays; `env_sensitive`
count by protocol (top offenders = protocols with hidden dependencies).

**Risks.** Minimal — fingerprinting is passive; wrong fingerprints cannot
falsify anything, only annotate.

## 4. Federated knowledge exchange — signed evidence bundles between labs

**What.** New owner CLI pair:
* `agentics export-knowledge --out bundle.json` — verified_knowledge +
  settled evidence + protocol_code + criterion + provenance, content-hashed;
* `agentics import-knowledge bundle.json` — **never trusts imported
  verdicts**: every imported claim is pre-registered as `proposed`
  (provenance `foreign:<lab-id>`) and immediately queued through the P23
  replay path; only a LOCAL re-run settles it (`foreign_claim_reconfirmed`
  / `foreign_claim_refuted` journal events).

**Why advanced.** Federation of self-verifying agent labs — H3 ("share
results so every AI knows") extended across machine and repo boundaries,
while keeping the lab's core rule intact: an exit code run HERE is the
only truth. Concretely useful today: this repo is developed on a home
machine, a sandbox and a VPS at the same time; three ledgers can now
cross-pollinate without merging code histories.

**Integration.** `agentics/memory/knowledge_bundle.py` (export/import,
bundle schema v1, sha256 manifest), CLI commands (owner-only), import path
reuses `register_hypothesis` + `replay_settled` — no new verdict logic.

**KPI.** bundles exchanged; foreign-claim reconfirmation rate (a low rate
= the other lab's environment differs — see proposal #3).

**Risks.** Trust boundary: import stays a manual owner command, never an
agent action; bundles carry no credentials; schema is append-only so a
bad bundle cannot rewrite local history.

## 5. Auto-ablation engine — supported claims earn a causal mini-study

**What.** When a claim settles `supported`, the local router generates 1–2
ablation variants (remove the claimed mechanism component, or swap in a
naive baseline) that run through the SAME protocol gate; the measured
delta (effect with vs without the component) is stored on the hypothesis
row. PUCT Q and the Elo prior consume the delta — the swarm learns not
just THAT an idea works but WHICH COMPONENT carries the effect, and
future pairings/deep-dives anchor on the strongest components.

**Why advanced.** This upgrades verification from binary to causal
attribution — the exact mechanism ablation studies credit in the
Co-Scientist loop the lab already ported, but applied to EVERY settled
claim automatically, with exit-code truth and bounded cost.

**Integration.** `agentics/verify/ablation.py`; settle-path hook in
`verify/experiment.py` (post-settlement, rate-gated); `hypotheses.ablation_delta`
column; journal `ablation_completed {delta, baseline, variant}`.

**KPI.** share of supported claims with a delta; average |delta|; do
high-|delta| claims reach merge faster (PUCT prior effect)?

**Risks.** Doubles protocol cost for supported claims — gated by
`ABLATION_RATE` (default 0 = today; recommended 0.3) and skip when the
protocol declares itself non-decomposable (no baseline contract line).

---

## Sequencing recommendation

1. **#3 hermetic replay first** — smallest effort, protects the ledger
   the lab already trusts, and unblocks #4 (meaningful cross-machine
   replication).
2. **#1 best-of-N second** — immediate quality lift on every code path
   using only existing gates; free when the local edge is up.
3. **#5 ablation** once #1 is stable (it consumes the sampler for variant
   generation).
4. **#4 federation** when a second machine is actually available to pair
   with (it is: sandbox + VPS + home).
5. **#2 Z3 last** — highest epistemic payoff, highest design cost; start
   with a thin `kind=invariant` pilot on arithmetic bounds before
   widening.

## Alternates considered (not picked for wave 3)

* **WASM/nsjail hermetic sandbox for protocol execution** — real
  isolation win, but infrastructure hardening rather than a research
  capability; revisit as an ops wave alongside sandbox/integrity.
* **Semantic LLM response cache (embedding-keyed)** — pure cost saver;
  the P20 budget + pacing already bound spend.
* **Curriculum scheduling for the frontier** — EIG (wave-2 #3) already
  ranks by information-per-cost; a curriculum layer would duplicate it.
