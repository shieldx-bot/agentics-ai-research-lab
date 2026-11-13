"""P16 (research side) — empirical verification for RESEARCH claims.

Composes the predecessor's stack instead of duplicating it:

    verify_ops      — the Coder path: plan-declared verify_cmds run for real,
                      verdicts from exit codes, evidence recorded (P16)
    verify_metrics  — pre-registered Criterion + AGENTICS_RESULT protocol
                      contract + 2-run reproducibility + machine adjudication
    shared memory   — `evidence` table (broadcast verdicts, H3) and
                      `hypotheses` table (pre-registration lifecycle:
                      proposed -> testing -> supported|refuted|inconclusive)

This package adds the piece that was missing from the live cycle: the
RESEARCHER path. node_research used to end in LLM prose; with
hypothesis.py + experiment.py the researcher states a falsifiable claim,
pre-registers it, runs a protocol in its own workspace, and settles the
claim into shared verified knowledge every agent's prompt can see.
"""
from agentics.verify.experiment import (
    register_hypothesis,
    run_hypothesis,
    settle_hypothesis,
)
from agentics.verify.hypothesis import (
    ALLOWED_OPS,
    EXPERIMENT_DIR,
    HYPOTHESIS_PROMPT_CONTRACT,
    PROTOCOL_CONTRACT,
    Hypothesis,
    claim_hash,
    hypothesis_from_json,
    parse_hypothesis,
    slugify,
    stable_id,
)

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
    "register_hypothesis",
    "run_hypothesis",
    "settle_hypothesis",
]
