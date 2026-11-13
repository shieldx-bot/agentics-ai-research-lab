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
> `inconclusive` and can never pose as success.
>
> **H3 — Can an agent share its verification results with every AI?**
> Yes — verdicts land in the shared `evidence` + `hypotheses` tables,
> auto-broadcast to the swarm (messages + journal), rendered on the owner's
> dashboard, queryable via `agentics evidence` / `agentics hypotheses`, and
> injected into EVERY agent's prompt as settled knowledge (supported facts
> become reusable common ground; refuted claims stop being repeated).

The stack grew from several parallel contributors on this branch; this
document describes how the pieces compose today.

---

## 1. The pieces and who runs them

| layer | module | who triggers it | verdict source |
| --- | --- | --- | --- |
| declared checks (MR flow) | `verify_ops.py` (`run_experiments`) | Coder/Writer: `verify_cmds` declared in `node_plan`, executed in `node_implement` before push | process exit code (`confirmed` / `refuted` / `inconclusive`) |
| hypothesis contract (MR flow) | `agentics/verify/` (`hypothesis.py`, `experiment.py`: register_hypothesis → run_hypothesis → settle_hypothesis) | Coder/Writer: `hypotheses` declared in `node_plan` (claim + prediction + criterion + experiment code) | measurement vs criterion, re-adjudicated from the protocol's `AGENTICS_RESULT` output, 2-run reproducibility gate |
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

## 6. Owner surfaces

```bash
python -m agentics.cli evidence            # every recorded run (P16)
python -m agentics.cli hypotheses          # pre-registered claims + machine verdicts (P16bis)
python -m agentics.cli dashboard           # live panels: evidence + hypotheses
```

On the dashboard, the "Kiểm chứng thực nghiệm" card shows raw runs; the
"Giả thuyết đã đăng ký" card shows each claim's criterion, the measured
value and the settled status — pre-registration means the goalposts cannot
move after seeing the numbers.

## 7. Testing

`tests/test_verify_ops.py` (exit-code evidence), `tests/test_verify_metrics.py`
(criterion adjudication + reproducibility), `tests/test_evidence_memory.py`
(shared ledger), `tests/test_hypothesis_memory.py` (lifecycle + verified
knowledge), `tests/test_node_verify.py` (researcher flow node-level),
`tests/test_p16_bridge.py` (plan-declared hypotheses), plus the demo:
`python scripts/local_mode_demo.py` prints a `supported` and a `refuted`
hypothesis from real protocol executions.
