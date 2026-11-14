"""Local Mode end-to-end demo — the full swarm lifecycle without GitHub.

Runs the REAL graph, REAL SQLite tasks/reviews, REAL bare repo (data/lab.git)
with its pre-receive hook, REAL P10 guard layers, REAL tier-2 verification
and REAL local merges. The only stub is the LLM edge (deterministic offline
answers, so the demo needs no cookies and no network).

What it shows, in order:
  1. the OWNER queues tasks (+ the seeded starter backlog);
  2. the SUPERVISOR scans, sweeps stale claims, tops the backlog up;
  2b. the RESEARCHER researches, then SELF-VERIFIES its finding (P16bis):
      claims are pre-registered with a quantitative criterion, protocols are
      written into experiments/ and RUN twice in the guarded workspace, the
      machine adjudicates measured-vs-criterion and the verdicts broadcast
      to every agent (owner hypotheses 1-3, verified end to end);
  3. the CODER claims a task, researches, plans, implements, pushes a branch
     and opens an MR (the P10 guard and the pre-receive hook watch every push);
  4. the WRITER does the same for a docs task;
  5. the protection demo: an agent tries to edit agentics/stop.py and is
     stopped by guard layer 1 (write-time) before anything leaves the workspace;
  6. the REVIEWERS cross-review both MRs as a P19 PANEL: two peers vote
     accept each MR (quorum = panel size 2) -> unanimous approval;
  7. the INTEGRATOR drains the queue: tier-2 (full tests on main+MR), then a
     local merge into data/lab.git:main (P4);
  8. the journal timeline + live state that make the swarm observable.

Usage:
    python scripts/local_mode_demo.py [--keep]   # --keep: keep the data dir
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.agents.base import AgentContext  # noqa: E402
from agentics.agents.roles import Role  # noqa: E402
from agentics.config import Settings  # noqa: E402
from agentics.github_ops.repo_workspace import RepoWorkspace  # noqa: E402
from agentics.graph.research_graph import run_cycle  # noqa: E402
from agentics.local_ops import (  # noqa: E402
    LocalClaimPolicy, LocalMergeQueue, LocalOps,
)
from agentics.memory.personal_memory import PersonalMemory  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.memory.vector_store import VectorStore  # noqa: E402
from agentics.policies.ownership import OwnershipPolicy  # noqa: E402

TOPIC = "ML for early failure prediction of network functions in NFV"


def _resp(answer: str, citations: list[str] | None = None):
    """LLMResponse-shaped namespace (papers.py checks ``resp.ok``)."""
    return SimpleNamespace(answer=answer, citations=citations or [], ok=bool(answer))


class FakeLLMHost:
    """Deterministic offline stand-in for BrowserSessionHost."""

    account = SimpleNamespace(name="offline-demo")

    async def ask(self, query, *, mode="concise", system=None, purpose="general"):
        if purpose == "research":
            answer = (
                f"Survey on {TOPIC}: sliding-window telemetry features feed "
                "gradient boosting and LSTM models; Random Forest reaches ~93% "
                "precision predicting VM failures minutes ahead. "
                "[demo answer — offline]"
            )
            return _resp(answer, ["https://arxiv.org/demo/1", "https://ieee.demo/2"])
        if purpose == "plan":
            path = self._path_for(query)
            return _resp(json.dumps({
                "files": [{"path": path, "change": "new"}],
                "steps": [f"create {path} from the latest findings"],
                "commit_message": f"add {Path(path).name}",
                "reuses": [], "reusable_tools": [],
            }))
        if purpose == "review":
            return _resp(json.dumps({"verdict": "approve",
                                     "comment": "demo review: changes look good"}))
        if purpose == "verify":
            # P16bis: 2 claims, each with a self-contained protocol. The first
            # measures precision ~0.9 (criterion >= 0.9 -> supported); the
            # second measures 0.5 (refuted). The verdict comes from the
            # machine comparison, not from this fake LLM.
            good = (
                "import json\n"
                "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
                "print('toy simulation: 200 predictions')\n"
                "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
            )
            bad = (
                "import json\n"
                "correct = sum(1 for i in range(200) if (i * 7) % 10 < 4)\n"
                "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
            )
            return _resp(json.dumps({"hypotheses": [
                {"claim": "ensemble voting reaches >=0.9 precision on the toy set",
                 "metric": "precision", "op": ">=", "target": 0.9,
                 "protocol": good},
                {"claim": "single-tree baseline reaches >=0.9 precision on the toy set",
                 "metric": "precision", "op": ">=", "target": 0.9,
                 "protocol": bad},
            ]}))
        # implement: return the full content of the file the plan mentions
        m = re.search(r'"path":\s*"([^"]+)"', query)
        path = m.group(1) if m else "docs/untitled.md"
        if path.endswith(".md"):
            content = (f"# Research note (demo)\n\nTopic: {TOPIC}\n\n"
                       "Gradient boosting + LSTM lead on the 5-10 min horizon.\n")
        else:
            content = ('"""Demo regression test (local mode demo)."""\n\n'
                       "def test_demo_pipeline():\n    assert 1 + 1 == 2\n")
        return _resp(content)

    @staticmethod
    def _path_for(plan_prompt: str) -> str:
        head = (plan_prompt.split("Issue #", 1)[1][:200].lower()
                if "Issue #" in plan_prompt else "")
        if re.search(r"\bdoc(s|ument)", head):
            return "docs/local_demo_findings.md"
        return "tests/test_demo_local.py"


def build_context(agent_id: str, role: Role, settings: Settings, mem: SharedMemory,
                  vector: VectorStore, local: LocalOps,
                  ownership: OwnershipPolicy) -> AgentContext:
    return AgentContext(
        agent_id=agent_id, role=role, settings=settings, memory=mem, vector=vector,
        llm=FakeLLMHost(),
        workspace=RepoWorkspace(agent_id, settings),
        ownership=ownership,
        local=local,
        claims=LocalClaimPolicy(mem, settings.claim_ttl_minutes),
        # demo intent: show the FULL lifecycle including the merge itself
        # (production default keeps AUTO_MERGE off — the owner merges manually)
        merge_queue=LocalMergeQueue(mem, auto_merge=True, guard=local.guard),
        meta={"account": "offline-demo",
              "personal": PersonalMemory(agent_id, settings.data_dir / "personal")},
    )


def show(mem: SharedMemory, title: str) -> None:
    print(f"\n=== {title} ===")
    counts = mem.task_counts()
    print("tasks: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    for mr in mem.recent_reviews(limit=4):
        print(f"  MR #{mr['id']} [{mr['status']}] {mr['title'][:56]!r} (by {mr['author']})")
    for a in mem.active_agents(stale_after_s=3600):
        goal = (a.get("intent") or {}).get("goal", "-")
        node = a.get("current_node") or "-"
        print(f"  {a['agent']} {a['role']:<11} cycle {a['cycle']:<3} "
              f"{a['status']:<9} node={node:<10} intent: {str(goal)[:56]}")


async def main() -> None:
    global TOPIC
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the demo data dir")
    ap.add_argument("--topic", default=TOPIC,
                    help="research topic for the demo (default: the built-in NFV one)")
    args = ap.parse_args()
    TOPIC = args.topic

    root = Path(tempfile.mkdtemp(prefix="agentics-demo-"))
    print(f"demo sandbox: {root}")
    print(f"demo topic: {TOPIC}")
    settings = Settings(data_dir=root / "data", research_topic=TOPIC,
                        evolution_every_n_cycles=0)
    settings.ensure_dirs()

    mem = SharedMemory(root / "memory.db")
    local = LocalOps(mem, settings)
    # seed the bare repo from a tiny fake checkout (keeps the demo self-contained)
    src = root / "seed"
    src.mkdir()
    import subprocess
    subprocess.run(["git", "init", "-b", "main", str(src)], check=True,
                   capture_output=True)
    (src / "agentics").mkdir()
    (src / "agentics" / "stop.py").write_text("# real lab code stands here\n")
    (src / "docs").mkdir()
    (src / "docs" / "README.md").write_text("lab docs\n")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=owner@lab",
                    "-c", "user.name=owner", "commit", "-m", "seed lab"],
                   check=True, capture_output=True)
    local.repo.ensure(src=src)
    # owner queues the work FIRST — ensure_backlog() then leaves it untouched
    code_task = mem.create_task(
        "Add a demo regression test for the research-note pipeline",
        body="Small runnable test proving the note pipeline works.", agent="owner")
    docs_task = mem.create_task(
        f"Document local findings: {TOPIC}",
        body="Turn the latest findings into a research note.", agent="owner")
    local.tasks.ensure_backlog()      # no-op: the backlog is not empty
    print(f"local bare repo ready: {local.repo.path}")
    print(f"owner added tasks #{code_task} (code) and #{docs_task} (docs)")

    ownership = OwnershipPolicy()
    supervisor = build_context("a01", Role.SUPERVISOR, settings, mem,
                               VectorStore(settings.chroma_path), local, ownership)
    researcher = build_context("a02", Role.RESEARCHER, settings, mem,
                               supervisor.vector, local, ownership)
    coder = build_context("a03", Role.CODER, settings, mem, supervisor.vector,
                          local, ownership)
    writer = build_context("a06", Role.WRITER, settings, mem, supervisor.vector,
                           local, ownership)
    reviewer = build_context("a07", Role.REVIEWER, settings, mem,
                             supervisor.vector, local, ownership)
    # P19 needs a PANEL: quorum accept votes (default 2) from DIFFERENT
    # agents, and P24bis spends a reviewer cycle on replication first — one
    # lone reviewer can never reach quorum, so the demo fields two.
    reviewer2 = build_context("a08", Role.REVIEWER, settings, mem,
                              supervisor.vector, local, ownership)
    integrator = build_context("a09", Role.INTEGRATOR, settings, mem,
                               supervisor.vector, local, ownership)

    # 2) supervisor scan -----------------------------------------------------
    await run_cycle(supervisor)
    show(mem, "after supervisor scan (backlog topped up)")

    # 2b) researcher cycle: research -> verify (P16bis) -----------------------
    await run_cycle(researcher)
    show(mem, "after researcher cycle (finding verified by RUNNING protocols)")
    print("  hypotheses:")
    for h in mem.list_hypotheses(limit=5):
        c = h["criterion"]
        print(f"    h#{h['id']} [{h['status']}] {h['claim'][:64]!r} "
              f"criterion: {c.get('metric')} {c.get('op')} {c.get('target')} "
              f"— {h['verdict_note'][:70]}")
    print(f"  verified knowledge now in every agent's prompt: "
          f"{len(mem.verified_knowledge())} settled claim(s)")

    # 3) coder full cycle -----------------------------------------------------
    await run_cycle(coder)
    show(mem, "after coder cycle (branch pushed, MR opened)")

    # 4) protection demonstration ---------------------------------------------
    evil_state = {
        "branch": "agent/a03/task-9-evil", "issue_number": code_task,
        "issue_title": "hack the lab", "issue_body": "", "category": "code",
        "plan": json.dumps({"files": [{"path": "agentics/stop.py",
                                       "change": "edit"}],
                            "steps": ["tamper"],
                            "commit_message": "evil"}),
        "papers": [{"answer": ""}],
    }
    coder.workspace.sync_main()
    coder.workspace.checkout_new_branch(evil_state["branch"])
    from agentics.graph import nodes as n
    result = await n.node_implement(coder, evil_state)
    print(f"\nPROTECTION GUARD: plan touching agentics/stop.py -> yield_reason: "
          f"{result.get('yield_reason', '(none)')!r}")
    coder.workspace.sync_main()          # discard the tampered tree

    # 5) writer cycle ----------------------------------------------------------
    await run_cycle(writer)
    show(mem, "after writer cycle")

    # 6) reviewers cross-review both MRs (P5) + replicate settled claims (P24bis)
    #    Rounds alternate the two reviewers until BOTH panels reach quorum.
    for _ in range(4):
        await run_cycle(reviewer)
        await run_cycle(reviewer2)
    show(mem, "after reviews (panel voted both MRs)")

    # 7) integrator drains the queue ----------------------------------------------
    await run_cycle(integrator)
    show(mem, "after integrator (merged into main)")

    # 8) journal timeline ------------------------------------------------------------
    print("\n=== journal timeline (owner view) ===")
    for e in mem.recent_journal(limit=40):
        detail = e.get("detail")
        detail = json.dumps(detail, ensure_ascii=False)[:60] if isinstance(detail, dict) else str(detail)[:60]
        print(f"  [{e['agent']}] {e['event']:<15} {detail}")

    log = subprocess.run(
        ["git", "-C", str(local.repo.path), "log", "--oneline", "-6", "main"],
        capture_output=True, text=True).stdout.strip()
    print(f"\nmain history on the local bare repo:\n{log}")

    for ctx in (supervisor, researcher, coder, writer, reviewer, reviewer2,
                integrator):
        ctx.meta["personal"].close()
    mem.close()

    if args.keep:
        print(f"\ndata kept at {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
        print("\ndemo sandbox cleaned up")

    await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
