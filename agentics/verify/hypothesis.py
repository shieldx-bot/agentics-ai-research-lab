"""Hypothesis — the contract that turns a research claim into a runnable test.

A finding used to end as LLM prose in shared memory. A *hypothesis* is the
missing piece of a real researcher's workflow: a claim, the prediction it
makes, ONE measurable success criterion, and the Python source of an
experiment that can decide it. The experiment source is written by the agent
(LLM) but it never gets the last word: `runner.py` executes it and
re-evaluates the criterion from the raw evidence.

The JSON contract the LLM must answer with (kept beside the parser so the
prompt and the validator cannot drift apart) is `HYPOTHESIS_PROMPT_CONTRACT`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# The ONLY comparison operators a criterion may use (kept small on purpose:
# every operator here is re-evaluated mechanically in runner.evaluate_criteria).
COMPARISONS: tuple[str, ...] = ("<=", ">=", "==", ">", "<")

MAX_CODE_CHARS = 20_000        # experiment source cap (workspace file, not a log)
MAX_CLAIM_CHARS = 1_000
MAX_PREDICTION_CHARS = 1_000
MAX_METRIC_CHARS = 120
SLUG_RE = re.compile(r"[^a-z0-9_-]+")

HYPOTHESIS_PROMPT_CONTRACT = (
    '{"claim": "<the assertion to test, one sentence>", '
    '"prediction": "<the observable outcome if the claim holds>", '
    '"metric": "<name of the single measured quantity>", '
    '"comparison": "<= | >= | == | > | <", '
    '"threshold": <numeric success threshold>, '
    '"code": "<FULL python source of the experiment>"}'
)


@dataclass
class Hypothesis:
    """One testable assertion + its experiment (P16 unit of work)."""

    hypothesis_id: str            # slug — names the script and the result file
    claim: str
    prediction: str
    metric: str
    comparison: str
    threshold: float
    code: str                     # full python source of the experiment
    source_finding: int | None = None   # finding the claim came from
    task: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def script_path(self) -> str:
        """Where the experiment lives inside the agent workspace (P3: the
        researcher's own territory — no ownership conflicts with the Coder)."""
        return f"research_notes/experiments/{self.hypothesis_id}.py"

    @property
    def result_path(self) -> str:
        """Where the experiment MUST write its evidence JSON (runner-read)."""
        return f"research_notes/experiments/results/{self.hypothesis_id}.json"

    def to_json(self) -> str:
        return json.dumps(
            {
                "hypothesis_id": self.hypothesis_id,
                "claim": self.claim,
                "prediction": self.prediction,
                "metric": self.metric,
                "comparison": self.comparison,
                "threshold": self.threshold,
                "code": self.code,
                "source_finding": self.source_finding,
                "task": self.task,
            },
            ensure_ascii=False,
        )


def slugify(text: str, fallback: str = "hypothesis") -> str:
    """Lowercase slug safe for a workspace filename, 1-48 chars."""
    slug = SLUG_RE.sub("-", (text or "").lower()).strip("-")[:48]
    return slug or fallback


def _clean_text(raw: object, max_chars: int) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip()[:max_chars]


def _as_number(raw: object) -> float | None:
    """Accept int/float (bool is NOT a number here) or a numeric string."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None


def parse_hypothesis(raw: dict) -> tuple[Hypothesis | None, str]:
    """Validate an LLM-proposed hypothesis dict -> (Hypothesis | None, reason).

    Anything malformed returns (None, reason) — the caller journals it and the
    cycle degrades gracefully instead of guessing. The verdict itself is never
    decided here; this only gates WHAT is allowed to become an experiment.
    """
    if not isinstance(raw, dict):
        return None, "hypothesis payload is not an object"

    claim = _clean_text(raw.get("claim"), MAX_CLAIM_CHARS)
    prediction = _clean_text(raw.get("prediction"), MAX_PREDICTION_CHARS)
    metric = _clean_text(raw.get("metric"), MAX_METRIC_CHARS)
    comparison = str(raw.get("comparison", "")).strip()
    code = raw.get("code")
    threshold = _as_number(raw.get("threshold"))

    if not claim:
        return None, "missing claim"
    if not prediction:
        return None, "missing prediction"
    if not metric:
        return None, "missing metric"
    if comparison not in COMPARISONS:
        return None, f"comparison must be one of {', '.join(COMPARISONS)}"
    if threshold is None:
        return None, "threshold must be a number"
    if not isinstance(code, str) or not code.strip():
        return None, "missing experiment code"
    if len(code) > MAX_CODE_CHARS:
        return None, f"experiment code exceeds {MAX_CODE_CHARS} chars"

    hyp_id = slugify(str(raw.get("hypothesis_id") or claim))
    finding = raw.get("source_finding")
    task = raw.get("task")
    return (
        Hypothesis(
            hypothesis_id=hyp_id,
            claim=claim,
            prediction=prediction,
            metric=metric,
            comparison=comparison,
            threshold=threshold,
            code=code.strip(),
            source_finding=int(finding) if isinstance(finding, int) else None,
            task=int(task) if isinstance(task, int) else None,
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
