"""P16-R — bounded self-repair for inconclusive hypothesis protocols.

The Reflexion move (Shinn et al., 2023) applied to the lab's own weakest
spot: the machine verdict was MECHANICALLY CORRECT (exit codes, metric
parsing, reproducibility gate) while the lab kept settling hypotheses
`inconclusive` because the agent's self-written protocols broke the
AGENTICS_RESULT contract — and the failure output was thrown away instead
of being read back into the system.

What this module does (pure parts — the LLM glue lives in graph.nodes):

* `build_repair_prompt` — feeds the author its OWN protocol, the FROZEN
  pre-registered criterion and the machine failure note back, and asks
  for a full corrected file. The criterion is never shown as editable:
  the goalposts cannot move after registration.
* `extract_fixed_code` — accepts fenced or bare replies, rejects empty
  or UNCHANGED code (a no-op repair must not burn a protocol re-run).

The verdict after a repair round is computed by exactly the same machine
adjudication as before (run_protocol / run_hypothesis). Repair improves
the SCRIPT quality; it never touches the verdict logic — that is the
honesty line between "self-improvement" and "metric gaming".
"""
from __future__ import annotations

from agentics.verify_metrics import PROTOCOL_CONTRACT, Criterion
from agentics.verify_ops import Evidence

REPAIR_INSTRUCTION = (
    "Your protocol did NOT produce a machine-parsable measurement, so the "
    "hypothesis was recorded inconclusive. Fix YOUR OWN protocol below so it "
    "satisfies the contract and measures the criterion's metric. Keep it a "
    "single self-contained python file, standard library only, deterministic.\n\n"
    "Return the FULL corrected file content only, no explanations."
)


def build_repair_prompt(claim: str, criterion: Criterion, code: str,
                        ev: Evidence) -> str:
    """Everything the author needs to fix its own protocol, bounded."""
    note = ""
    meta = getattr(ev, "meta", None) or {}
    if isinstance(meta, dict):
        note = str(meta.get("note") or "")
    return (
        f"Hypothesis (pre-registered, criterion FROZEN — do not change it):\n"
        f"{claim[:400]}\n\n"
        f"Criterion: {criterion.describe()}\n\n"
        f"{PROTOCOL_CONTRACT}\n"
        f"Why the machine rejected the run: {note[:400] or ev.verdict}\n"
        f"Process output tail:\n{(ev.output or '')[-800:]}\n\n"
        f"Your current protocol:\n```python\n{code[:6000]}\n```\n\n"
        f"{REPAIR_INSTRUCTION}"
    )


def extract_fixed_code(answer: str, old_code: str) -> str | None:
    """The corrected file from the reply, or None.

    Strips one fenced block (```python ... ``` or ``` ... ```); rejects
    empty replies and byte-identical rewrites (no-op repair)."""
    if not answer or not answer.strip():
        return None
    text = answer.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.rstrip()
    if not text.strip():
        return None
    if text.strip() == (old_code or "").strip():
        return None
    return text
