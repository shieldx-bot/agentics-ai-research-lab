"""P16 — Empirical verification: agents judge ideas by running code, not by
trusting an LLM's prose.

    hypothesis.py — the contract: claim + prediction + measurable criterion +
                    the experiment source (what the agent proposes to run)
    runner.py     — the executor: runs the experiment inside the agent's
                    workspace through the existing P2-guarded run_command,
                    then computes the verdict FROM THE EVIDENCE (exit code +
                    result JSON re-evaluated here), never from what the
                    script (or an LLM) claims about itself.

Verdicts: confirmed | refuted | inconclusive. Every run lands in the shared
`verifications` ledger (shared memory) so ALL agents can read, quote and
re-run each other's experiments — the swarm-level equivalent of publishing
results instead of opinions.
"""
from agentics.verify.hypothesis import (
    COMPARISONS,
    HYPOTHESIS_PROMPT_CONTRACT,
    Hypothesis,
    hypothesis_from_json,
    parse_hypothesis,
)
from agentics.verify.runner import (
    RESULT_DIR,
    ExperimentRunner,
    VerificationResult,
    evaluate_criteria,
)

__all__ = [
    "COMPARISONS",
    "HYPOTHESIS_PROMPT_CONTRACT",
    "Hypothesis",
    "hypothesis_from_json",
    "parse_hypothesis",
    "RESULT_DIR",
    "ExperimentRunner",
    "VerificationResult",
    "evaluate_criteria",
]
