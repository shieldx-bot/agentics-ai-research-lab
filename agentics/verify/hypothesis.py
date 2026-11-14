"""P16 (research side) — the hypothesis contract that composes the
predecessor's verification stack:

    verify_metrics.Criterion  — the pre-registered acceptance test (unchanged;
                                this module REUSES it instead of re-inventing
                                a second adjudicator)
    verify_metrics.run_protocol — the executor + reproducibility gate (2 runs,
                                AGENTICS_RESULT marker, machine verdict)
    shared memory hypotheses  — the pre-registration ledger + lifecycle
                                (proposed -> testing -> supported | refuted |
                                inconclusive)

What was missing before this module: a *research-side* path. The Coder path
(plan declares verify_cmds) already exists; but a Researcher's findings ended
as LLM prose — no claim was ever stated precisely enough to be testable, and
nothing called add_hypothesis / run_protocol / verified_knowledge in the live
cycle. This module defines the ONE JSON contract the LLM answers with, its
mechanical validation, and the register -> run -> settle glue.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from agentics.verify.falsify import FalsificationSpec
from agentics.verify_metrics import PROTOCOL_CONTRACT, Criterion

# Researcher-owned territory (P3 ownership + writable_roots): experiments and
# their evidence live under research_notes/, so no ownership conflict with the
# Coder's verify_cmds checks.
EXPERIMENT_DIR = "research_notes/experiments"
# The hypothesis contract: the protocol WRITES its measurement here, the
# runner reads it back and adjudicates (port of the 0086494 contract that the
# bridge tests pin; 127862f deleted the runner but the contract stayed).
RESULT_DIR = "research_notes/experiments/results"

MAX_CODE_CHARS = 20_000
MAX_CLAIM_CHARS = 500          # hypotheses.claim column caps at 500
MAX_PREDICTION_CHARS = 1_000

# Criterion ops come from the predecessor's adjudicator — one vocabulary.
ALLOWED_OPS = ("<=", ">=", "<", ">", "==", "!=", "between")

SLUG_RE = re.compile(r"[^a-z0-9_-]+")

# The exact JSON the hypothesize node asks the LLM for (kept beside the
# validator so prompt and parser cannot drift).
HYPOTHESIS_PROMPT_CONTRACT = (
    '{"claim": "<the assertion to test, ONE sentence, falsifiable>", '
    '"prediction": "<the observable outcome if the claim holds>", '
    '"criterion": {"metric": "<name of the single measured quantity>", '
    '"op": "<= | >= | < | > | == | != | between", "target": <number>'
    '[, "tolerance": <upper bound, only for between>]}, '
    '"code": "<FULL python source of the protocol>"}'
)


@dataclass
class Hypothesis:
    """One testable research claim + its pre-registered criterion + protocol.

    The verdict is NEVER stored here — it belongs to the shared hypotheses /
    evidence tables once the protocol has actually run (H2: evidence over
    opinion)."""

    hypothesis_id: str            # slug — names the protocol script
    claim: str
    prediction: str
    criterion: Criterion
    code: str                     # full python source of the protocol
    source_finding: int | None = None
    task: int | None = None
    falsification: "FalsificationSpec | None" = None   # P25: the declared
    # input class X + metamorphic relations + mutation gate. None = the
    # claim is verified by the base protocol alone (the default — the P16
    # contract is unchanged for every caller that does not declare it).

    @property
    def script_path(self) -> str:
        return f"{EXPERIMENT_DIR}/{self.hypothesis_id}.py"

    @property
    def result_path(self) -> str:
        """Where the protocol may write its measured value (RESULT-FILE
        convention: {"value": <number>}); the runner falls back to it when
        the script prints no AGENTICS_RESULT marker."""
        return f"{EXPERIMENT_DIR}/results/{self.hypothesis_id}.json"

    @property
    def falsifies(self) -> bool:
        """Whether the claim declared a falsification suite (P25)."""
        return self.falsification is not None

    # Flat aliases over the criterion — the plan contract (node_plan) speaks
    # {metric, comparison, threshold}; the criterion keeps the canonical form.
    @property
    def metric(self) -> str:
        return self.criterion.metric

    @property
    def comparison(self) -> str:
        return self.criterion.op

    @property
    def threshold(self) -> float:
        return self.criterion.target

    def to_json(self) -> str:
        return json.dumps(
            {
                "hypothesis_id": self.hypothesis_id,
                "claim": self.claim,
                "prediction": self.prediction,
                "criterion": self.criterion.as_dict(),
                "code": self.code,
                "source_finding": self.source_finding,
                "task": self.task,
                "falsification": (self.falsification.as_dict()
                                  if self.falsification is not None else None),
            },
            ensure_ascii=False,
        )


def slugify(text: str, fallback: str = "hypothesis") -> str:
    slug = SLUG_RE.sub("-", (text or "").lower()).strip("-")[:40]
    return slug or fallback


def claim_hash(claim: str) -> str:
    """Stable identity of a claim (case/whitespace-insensitive).

    The dedup key that lets the swarm recognize an already-settled hypothesis
    however it is reworded: a supported or refuted claim is NOT re-run by the
    next agent that meets it — the first verdict IS the shared verdict (H3:
    never burn cycles re-verifying settled work)."""
    normalized = " ".join((claim or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_id(claim: str, seed: str = "") -> str:
    """Collision-free slug: readable prefix + short digest of the claim."""
    digest = hashlib.sha1(f"{claim}|{seed}".encode("utf-8")).hexdigest()[:8]
    return f"{slugify(claim)}-{digest}"


def _clean_text(raw: object, max_chars: int) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip()[:max_chars]


def parse_hypothesis(raw: object) -> tuple[Hypothesis | None, str]:
    """Validate an LLM-proposed hypothesis dict -> (Hypothesis | None, reason).

    Validation is mechanical (types, ops via the predecessor's Criterion,
    sizes); a malformed proposal is refused with a reason and journalled —
    never guessed through. Nothing here decides TRUE/FALSE."""
    if not isinstance(raw, dict):
        return None, "hypothesis payload is not an object"

    claim = _clean_text(raw.get("claim"), MAX_CLAIM_CHARS)
    prediction = _clean_text(raw.get("prediction"), MAX_PREDICTION_CHARS)
    code = raw.get("code")

    if not claim:
        return None, "missing claim"
    if not prediction:
        return None, "missing prediction"
    if not isinstance(code, str) or not code.strip():
        return None, "missing protocol code"
    if len(code) > MAX_CODE_CHARS:
        return None, f"protocol code exceeds {MAX_CODE_CHARS} chars"

    # The criterion may arrive in EITHER form:
    #   structured  — {"criterion": {"metric", "op", "target"[, "tolerance"]}}
    #   flat        — {"metric": ..., "comparison": ..., "threshold": ...}
    # The plan prompt (node_plan) and the P16 bridge contract emit the flat
    # form; the richer object form is accepted for callers that pre-compose a
    # Criterion (e.g. node_verify). Both are validated by the SAME
    # Criterion.from_dict — one vocabulary, one adjudicator.
    crit_raw = raw.get("criterion")
    if not isinstance(crit_raw, dict):
        # Flat plan-declared shape (node_plan's contract): {metric, comparison,
        # threshold} — the same criterion in a flatter coat; accept both so
        # prompt and validator cannot drift apart.
        if "metric" in raw and "comparison" in raw:
            crit_raw = {"metric": raw.get("metric"), "op": raw.get("comparison"),
                        "target": raw.get("threshold")}
        else:
            return None, "missing criterion object {metric, op, target}"

    try:
        criterion = Criterion.from_dict(crit_raw)
    except ValueError as exc:
        return None, f"invalid criterion: {exc}"
    if not criterion.metric:
        return None, "criterion metric is empty"

    finding = raw.get("source_finding")
    task = raw.get("task")
    hyp_id = slugify(str(raw.get("hypothesis_id") or claim))
    if len(hyp_id) < 8:                    # too short to stay unique
        hyp_id = stable_id(claim)
    # P25: the OPTIONAL falsification suite — parsed and validated with the
    # same mechanical strictness as the criterion (a malformed declaration
    # is refused with a reason, never guessed through).
    falsification: FalsificationSpec | None = None
    if raw.get("falsification") is not None:
        try:
            falsification = FalsificationSpec.from_dict(raw.get("falsification"))
        except ValueError as exc:
            return None, f"invalid falsification spec: {exc}"
    return (
        Hypothesis(
            hypothesis_id=hyp_id,
            claim=claim,
            prediction=prediction,
            criterion=criterion,
            code=code.strip(),
            source_finding=int(finding) if isinstance(finding, int) else None,
            task=int(task) if isinstance(task, int) else None,
            falsification=falsification,
        ),
        "",
    )


def hypothesis_from_json(text: str) -> tuple[Hypothesis | None, str]:
    """Parse the hypothesis JSON carried through the graph state."""
    try:
        raw = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        return None, f"hypothesis JSON invalid: {exc}"
    return parse_hypothesis(raw)


__all__ = [
    "ALLOWED_OPS",
    "EXPERIMENT_DIR",
    "HYPOTHESIS_PROMPT_CONTRACT",
    "PROTOCOL_CONTRACT",
    "Hypothesis",
    "claim_hash",
    "hypothesis_from_json",
    "parse_hypothesis",
    "slugify",
    "stable_id",
]
