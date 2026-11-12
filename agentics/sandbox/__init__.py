"""Sandbox guards — integrity of the running lab itself.

P10 changeset protection (agentics/policies/guard.py) stops AGENT writes into
the engine at write time, diff time and merge time; this package adds the
DETECT layer: hashing the runtime tree the orchestrator actually runs from so
any out-of-band drift — a botched merge, an escaped process, a manual edit —
surfaces as an owner-visible alert (agentics/sandbox/integrity.py).
"""
from agentics.sandbox.integrity import IntegrityGuard, build_manifest, verify

__all__ = ["IntegrityGuard", "build_manifest", "verify"]
