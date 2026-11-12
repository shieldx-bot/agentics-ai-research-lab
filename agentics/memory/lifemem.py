"""LifeMem-inspired lifelong memory for the swarm (Qiu et al., arXiv:2609.12655).

LifeMem's problem is exactly what happens to our `traces` table as the lab runs
for weeks: an ever-growing, unorganized pile of trajectories where (a) retrieval
gets noisier and (b) useful experience is never abstracted, so nothing transfers
across tasks. Its answers, adapted here:

1. **Workflow clustering** — incoming traces are assigned to a *workflow
   cluster* (same category, geometrically similar "attempted" text) instead of
   living in one flat table. Similar joins, dissimilar creates a new cluster.
2. **MRL diversity constraint** — a cluster is split into two only when it is
   both large (> volume) *and* geometrically loose (Mean Resultant Length
   < tau, Eq. 3 of the paper). Splitting never fires on homogeneous clusters.
3. **Skill distillation** — each cluster's *merged* traces are distilled into
   one short, environment-invariant skill (what worked, what to avoid).
   Offline the distiller is a deterministic heuristic; an LLM can refine it.
4. **Dual-tier recall** — planning gets BOTH concrete traces (env-aligned
   exemplars) AND the distilled skills of clusters in this category
   (transferable abstractions), mirroring the paper's M_act = (trajs, skills).

Deliberate lab adaptations (NOT in the paper, documented in docs/MEMORY.md):
- embeddings are deterministic hashed bag-of-words (no model download, no
  network; sufficient for short "attempted" texts), so the whole module is
  testable offline;
- clusters are scoped by issue *category* first — our coarse analogue of
  "environment", which keeps retrieval interference low (the paper's core
  motivation) even with the cheap embedding;
- skills are distilled only from merged traces (verified success), while the
  most common failure is kept as a one-line "avoid" clause.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Callable, Sequence

from agentics.memory.shared_memory import SharedMemory

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

# A trace joins an existing cluster at this cosine similarity, else it opens one.
JOIN_THRESHOLD = 0.35
# LifeMem defaults: split when n > L and R < tau_mrl (paper: L=50, tau=0.95).
SPLIT_VOLUME = 50
SPLIT_TAU_MRL = 0.95


# ======================================================================
# Embedding & geometry (offline, deterministic)
# ======================================================================
def embed(text: str, dim: int = 64) -> list[float]:
    """Unit-length hashed bag-of-words vector. Deterministic across processes."""
    vec = [0.0] * dim
    for word in _WORD_RE.findall((text or "").lower()):
        vec[zlib.crc32(word.encode()) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def mean_resultant_length(vectors: Sequence[Sequence[float]]) -> float:
    """R = || (1/n) * sum(x_i) || over unit vectors (Eq. 3 of the paper).

    R -> 1: members point the same way (redundant templates);
    R -> 0: members are heterogeneous (a cluster worth splitting).
    """
    if not vectors:
        return 0.0
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]
    return math.sqrt(sum(x * x for x in mean))


def two_means_split(vectors: Sequence[Sequence[float]], rounds: int = 3) -> list[int]:
    """Deterministic 2-means: seeds are the farthest pair, then Lloyd rounds.
    Returns a 0/1 label per input vector (input order preserved)."""
    n = len(vectors)
    if n < 2:
        return [0] * n
    # seeds: first vector + the vector least similar to it (farthest pair)
    seed0 = vectors[0]
    seed1 = min(vectors, key=lambda v: cosine(v, seed0))
    labels = [0] * n
    for _ in range(rounds):
        for i, v in enumerate(vectors):
            labels[i] = 0 if cosine(v, seed0) >= cosine(v, seed1) else 1
        members0 = [v for v, lb in zip(vectors, labels) if lb == 0]
        members1 = [v for v, lb in zip(vectors, labels) if lb == 1]
        if not members0 or not members1:
            break
        seed0 = [sum(v[d] for v in members0) / len(members0) for d in range(len(seed0))]
        seed1 = [sum(v[d] for v in members1) / len(members1) for d in range(len(seed1))]
    return labels


# ======================================================================
# Clustering
# ======================================================================
def assign_trace(
    memory: SharedMemory, trace_id: int, category: str, text: str,
    threshold: float = JOIN_THRESHOLD,
) -> int | None:
    """Assign a trace to its category's nearest workflow cluster (or open one).

    Idempotent: a trace that already belongs to a cluster stays put, so calling
    this again after `on_merge` never re-clusters anything.
    Returns the cluster id (None only if `text` is empty).
    """
    existing = memory.trace_cluster(trace_id)
    if existing is not None:
        return existing
    if not (text or "").strip():
        return None
    vec = embed(text)
    best_id, best_sim = None, -1.0
    for cl in memory.list_clusters(category):
        members = [m["attempted"] for m in memory.cluster_members(cl["cluster_id"])
                   if m["attempted"]]
        if not members:
            continue
        centroid = [sum(v[d] for v in (embed(t) for t in members)) / len(members)
                    for d in range(len(vec))]
        sim = cosine(vec, centroid)
        if sim > best_sim:
            best_id, best_sim = cl["cluster_id"], sim
    if best_id is not None and best_sim >= threshold:
        cluster_id = best_id
    else:
        cluster_id = memory.create_cluster(category, text.strip()[:120])
    memory.set_trace_cluster(trace_id, cluster_id)
    return cluster_id


def maybe_split(
    memory: SharedMemory, cluster_id: int,
    volume: int = SPLIT_VOLUME, tau_mrl: float = SPLIT_TAU_MRL,
) -> bool:
    """LifeMem's adaptive partitioning: split iff n > volume AND R < tau_mrl."""
    members = [m["attempted"] for m in memory.cluster_members(cluster_id)
               if m["attempted"]]
    if len(members) <= volume:
        return False
    r = mean_resultant_length([embed(t) for t in members])
    if r >= tau_mrl:                      # homogeneous: splitting adds no value
        return False
    labels = two_means_split([embed(t) for t in members])
    category = memory.get_cluster(cluster_id)["category"]
    new_id = memory.create_cluster(category, memory.get_cluster(cluster_id)["description"] + " (split)")
    for m, lb in zip(memory.cluster_members(cluster_id), labels):
        if lb == 1:
            memory.set_trace_cluster(m["id"], new_id)
    return True


# ======================================================================
# Skill distillation
# ======================================================================
_DISTILL_PROMPT = (
    "Below are task workflows from one cluster ({n} successful runs, category "
    "{cat}){fail_note}. Distill ONE transferable skill: 2-3 short imperative "
    "sentences describing the workflow that works regardless of the specific "
    "task, plus one sentence on what to avoid. Reply with the skill only.\n\n"
    "{body}"
)


def _heuristic_skill(cat: str, merged: list[dict], failed: list[dict]) -> str:
    """Offline distillation: patterns from what actually merged + top failure."""
    steps = [t["attempted"] for t in merged[-3:]]
    files = Counter(
        f.split("/")[-1] for t in merged for f in (t["files"] or []) if f
    ).most_common(3)
    parts = [f"Workflow ({cat}): " + " Then ".join(s[:120] for s in steps)]
    if files:
        parts.append("Usually touches: " + ", ".join(f"{n} (x{c})" for n, c in files))
    if failed and failed[-1]["failure"]:
        parts.append("Avoid: " + failed[-1]["failure"][:160])
    return " ".join(parts)


def distill_skill(
    memory: SharedMemory, cluster_id: int,
    llm_ask: Callable[[str], str] | None = None,
) -> str | None:
    """Distill a cluster into one active skill. Only *merged* traces count as
    evidence of a working workflow; the most recent failure becomes the avoid
    clause. `llm_ask(prompt) -> str` upgrades the heuristic if provided."""
    members = memory.cluster_members(cluster_id)
    merged = [m for m in members if m["outcome"] == "merged" and m["attempted"]]
    if not merged:
        return None                     # never distill from unverified work
    failed = [m for m in members if m["outcome"] in ("failed", "rejected") and m["failure"]]
    cat = memory.get_cluster(cluster_id)["category"]
    if llm_ask is not None:
        fail_note = f" and {len(failed)} failure(s)" if failed else ""
        body = "\n".join(f"- {m['attempted'][:300]}" for m in merged[-8:])
        try:
            text = " ".join(llm_ask(_DISTILL_PROMPT.format(
                n=len(merged), cat=cat, fail_note=fail_note, body=body)).split())
        except Exception:               # noqa: BLE001 — heuristic is the fallback
            text = ""
        if not text:
            text = _heuristic_skill(cat, merged, failed)
    else:
        text = _heuristic_skill(cat, merged, failed)
    memory.save_skill(cluster_id, text[:800])
    return text


def on_merge(
    memory: SharedMemory, trace_ids: Sequence[int],
    llm_ask: Callable[[str], str] | None = None,
) -> list[int]:
    """After a PR merge: (re)place its traces into workflow clusters, split if
    the cluster got loose, and refresh the cluster's distilled skill.
    Returns the cluster ids that were (re)distilled."""
    touched: list[int] = []
    for tid in trace_ids:
        tr = memory.get_trace(tid)
        if tr is None:
            continue
        cid = assign_trace(memory, tid, tr["category"], tr["attempted"])
        if cid is None:
            continue
        maybe_split(memory, cid)
        if distill_skill(memory, cid, llm_ask=llm_ask) and cid not in touched:
            touched.append(cid)
    return touched


# ======================================================================
# Dual-tier recall (skills tier)
# ======================================================================
def skill_context(
    memory: SharedMemory, query: str, category: str | None = None,
    limit: int = 5, max_chars: int = 1200,
) -> str:
    """The second tier of M_act: distilled skills, category-matched first, then
    keyword overlap with the current task, then popularity. Returned skills get
    their `uses` bumped (the lab's analogue of LifeMem's skill activation)."""
    skills = memory.list_skills(category, limit=40)
    if not skills:
        return "(no distilled skills yet)"
    words = {w.lower() for w in _WORD_RE.findall(query or "")}
    scored = []
    for s in skills:
        hay = f"{s['text']} {s['description']}".lower()
        hits = sum(1 for w in words if w in hay)
        scored.append((hits, s["uses"], s["skill_id"], s))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    lines = []
    for hits, _uses, sid, s in scored[:limit]:
        memory.bump_skill_use(sid)
        text = s["text"] if len(s["text"]) <= 300 else s["text"][:299] + "…"
        lines.append(f"- [{s['category']}] {text}")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 1] + "…"
