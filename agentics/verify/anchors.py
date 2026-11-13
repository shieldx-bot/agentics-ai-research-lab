"""Verdict provenance — self_authored_ratio and the settle anchor law.

The epistemic axis this module enforces is NOT "algorithm vs AI": an
algorithmic scorer can be just as self-graded as an LLM judge, only more
sophisticated. The axis is: *who holds the pen that writes the rubric, and
is the rubric anchored to something outside the agent's reach?*

The one-question audit for every scoring path:

    "Among the inputs that DECIDE this verdict, does at least ONE exist
     that the agent did not author?"

If the answer is empty, the verdict is self-graded — whether f(x) is a
language model, a regression or sha256. This module turns that question
into a number and a law:

* ``VerdictProvenance`` — the standard machine-verdict inputs
  (protocol / target / data) are all agent-authored; EXTERNAL ANCHORS
  (citation / peer_judge / replication / owner_approval / pinned_data /
  ci_verdict) are inputs the agent did not author. ``ratio`` is the
  self-authored share of the verdict's inputs.
* ``apply_settle_law`` — a verdict may be recorded `supported` ONLY when
  at least one anchor exists (ratio < 1). An anchorless "supported" is
  downgraded to the new `unanchored` lifecycle status: the measurement
  happened and passed the criterion, but the verdict is not yet trusted —
  its path to `supported` is INDEPENDENT REPLICATION (P24), an owner
  approval, or external data.
* ``detect_pinned_data`` — the protocol declares
  ``# AGENTICS_ANCHOR: pinned_data <path> <sha256>``; the anchor holds
  only when that workspace file EXISTS and its sha256 matches. The agent
  can READ the anchor, not WRITE the anchor (editing the data file breaks
  the hash; editing the declaration breaks the match).

Everything here is dependency-free and pure (workspace reads aside) so it
runs in tests without an LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# The standard inputs of EVERY machine protocol verdict — all authored by
# the agent: the agent writes the protocol, chooses the target, and its
# code produces the measured data. Deterministic adjudication of these is
# still self-graded (the sophisticated kind): the comparison cannot lie,
# but every term of the comparison was written by the judged party.
AUTHORED_INPUTS = ("protocol", "target", "data")

# External anchor kinds — verdict inputs the agent did NOT author:
#   citation       — URLs returned by an external source (Perplexity
#                    citations); the agent did not author the URL.
#   peer_judge     — a second opinion from an INDEPENDENT account
#                    (different cookie identity than the author).
#   replication    — an independent agent re-ran the same stored protocol
#                    and the machine verdict agreed (P24).
#   owner_approval — the human owner explicitly accepted the claim
#                    (local-only mode escape hatch).
#   pinned_data    — the protocol measured against a hash-locked external
#                    dataset (agent can read the anchor, not write it).
#   ci_verdict     — the verdict came from CI running OUTSIDE the agent's
#                    sandbox (GitHub Actions) — reserved; wired when the
#                    coder path settles from CI artifacts.
ANCHOR_KINDS = ("citation", "peer_judge", "replication", "owner_approval",
                "pinned_data", "ci_verdict")

# Protocol-source declaration: an anchored dataset. Parsed from the SAME
# code that is executed — the agent cannot claim an anchor its protocol
# does not actually use.
PINNED_DATA_RE = re.compile(
    r"AGENTICS_ANCHOR:\s*pinned_data\s+(\S+)\s+([0-9a-fA-F]{64})")


@dataclass
class VerdictProvenance:
    """Input provenance of ONE verdict: what the agent authored vs what
    anchors it to something outside its reach."""

    authored: list[str] = field(
        default_factory=lambda: list(AUTHORED_INPUTS))
    anchors: list[str] = field(default_factory=list)

    def add_anchor(self, *kinds: str) -> None:
        for k in kinds:
            if k in ANCHOR_KINDS and k not in self.anchors:
                self.anchors.append(k)

    @property
    def ratio(self) -> float:
        """self_authored_ratio = authored inputs / all deciding inputs.

        1.0 means every input is agent-authored (self-graded, however
        deterministic); < 1.0 means at least one external anchor exists.
        An empty provenance counts as fully self-authored (nothing was
        measured)."""
        total = len(self.authored) + len(self.anchors)
        if total <= 0:
            return 1.0
        return round(len(self.authored) / total, 3)

    @property
    def anchored(self) -> bool:
        return bool(self.anchors)

    def anchors_json(self) -> str:
        return json.dumps(self.anchors, ensure_ascii=False)


def provenance_for_settle(
    *, citations: int = 0, peer_judge: bool = False, replicated: bool = False,
    owner_approved: bool = False, pinned_data: bool = False,
    ci_verdict: bool = False,
) -> VerdictProvenance:
    """Build the provenance of a machine settle from its anchor flags.

    `citations` is a COUNT of external URLs attached to the verdict's
    origin answer (Perplexity returns citations natively); any positive
    count adds the `citation` anchor."""
    prov = VerdictProvenance()
    if citations and citations > 0:
        prov.add_anchor("citation")
    if peer_judge:
        prov.add_anchor("peer_judge")
    if replicated:
        prov.add_anchor("replication")
    if owner_approved:
        prov.add_anchor("owner_approval")
    if pinned_data:
        prov.add_anchor("pinned_data")
    if ci_verdict:
        prov.add_anchor("ci_verdict")
    return prov


def apply_settle_law(status: str, prov: VerdictProvenance,
                     require_anchor: bool = True) -> tuple[str, bool]:
    """The settle law: `supported` requires self_authored_ratio < 1.

    Returns (status_after, downgraded). An anchorless `supported` becomes
    the new `unanchored` lifecycle status — the measurement PASSED the
    pre-registered criterion but the verdict rests entirely on inputs the
    agent authored, so it is not yet common ground. refuted / inconclusive
    verdicts pass through unchanged (the law guards OPTIMISM, not
    pessimism — a self-graded failure is still a failure)."""
    if require_anchor and status == "supported" and not prov.anchored:
        return "unanchored", True
    return status, False


def detect_pinned_data(workspace, code: str) -> bool:
    """True when the protocol declares a pinned dataset AND the declared
    file in the workspace hashes to the declared sha256.

    The declaration lives IN the protocol source (so it travels with the
    claim to replication), the DATA lives in the workspace. Editing the
    data breaks the hash; editing the declaration breaks the match; the
    agent therefore cannot forge this anchor without inventing a sha256
    collision."""
    for m in PINNED_DATA_RE.finditer(code or ""):
        path, want = m.group(1), m.group(2).lower()
        try:
            data = workspace.read_file(path)
        except Exception:  # noqa: BLE001 — unreadable workspace = no anchor
            continue
        if not data:
            continue
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
        if digest == want:
            return True
    return False


__all__ = [
    "ANCHOR_KINDS", "AUTHORED_INPUTS", "PINNED_DATA_RE", "VerdictProvenance",
    "apply_settle_law", "detect_pinned_data", "provenance_for_settle",
]
