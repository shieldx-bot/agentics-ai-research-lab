"""Owner-exempt agent-code work — the dead write gate + the .env leak.

Follow-up on the canonical local-mode line. Before this patch the owner
exemption ('agentics allow-protected') was broken in two ways:

* DEAD GATE — `allow_protected` unblocked only the PUSH (layer 4); the write
  gate (layer 1) refused to even create an agentics/ file, so an exempt
  self-evolution task could never produce any change;
* CREDENTIAL LEAK — `.env` and `STOP` were not in ALWAYS_PROTECTED, so an
  exempt diff could carry the token file or the kill switch through every
  gate to main.

Now: an owner-exempt task may write agent code (write gate allow flag,
diff/review/merge gates keep their blanket-exempt check) — but credentials
are refused everywhere: .env and STOP joined ALWAYS_PROTECTED alongside
accounts/ and configs/secrets.yaml.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard, is_credential_path


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


class FakeClaims:
    def __init__(self, deny=()):
        self.deny = set(deny)

    def try_claim(self, number, agent, branch):
        return number not in self.deny


class FakeWS:
    def __init__(self, tmp_path, changed):
        self.path = tmp_path
        self._changed = changed
        self.pushed = []
        self.written: list[str] = []
        self.refused: list[str] = []

    def sync_main(self):
        pass

    def checkout_new_branch(self, b):
        pass

    def checkout_branch_from_origin(self, b):
        pass

    def read_file(self, p):
        return ""

    def write_file(self, p, c, allow=False):
        # mirror the REAL RepoWorkspace.write_file gate
        guard = self.guard
        if guard is not None and guard.is_protected(p) and not allow:
            self.refused.append(p)
            raise ValueError(f"P10 protected path: '{p}' is lab code")
        self.written.append(p)

    guard = None

    def changed_files(self):
        return list(self._changed)

    def run_command(self, cmd, timeout=120):
        return 0, "ok"

    def commit_and_push(self, branch, msg, allow_protected=False):
        self.pushed.append(branch)
        return True


WRITABLE_ROOTS = (
    "research_notes/", "experiments/", "benchmarks/", "tests/",
    "docs/", "reports/", "README.md",
)


def _impl_ctx(mem, ws, role=Role.CODER, exempt=False, issue_id=1):
    if mem.get_task(issue_id) is None:
        mem.create_task("self-evolution: prompts", category="evolution",
                        labels="kind/evolution")
    if exempt:
        mem.set_task_exempt(issue_id, True, agent="owner")

    async def _ask(prompt, **kw):
        return SimpleNamespace(answer="x = 1\n", citations=[])

    return SimpleNamespace(
        agent_id="a04", role=role, memory=mem,
        claims=FakeClaims(), cycle=1,
        meta={"personal": FakePersonal()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=10),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        llm=SimpleNamespace(ask=_ask, account=SimpleNamespace(name="t")),
        local=SimpleNamespace(guard=ProtectedPathsGuard(writable_roots=WRITABLE_ROOTS)),
        workspace=ws,
        ownership=__import__("agentics.policies.ownership",
                             fromlist=["x"]).OwnershipPolicy(),
    )


def _impl_state(category="evolution", issue_id=1, exempt=False, files=()):
    files = list(files)
    return {
        "branch": f"agent/a04/task-{issue_id}-x",
        "issue_number": issue_id,
        "issue_title": "self-evolution: prompts",
        "category": category,
        "task_exempt": exempt,
        "expected_files": [f.get("path", "") for f in files if isinstance(f, dict)],
        "plan": json.dumps({"files": files, "commit_message": "m"}),
    }


# ------------------------------------------------------------ helpers
def test_credential_helpers():
    assert is_credential_path(".env")
    assert is_credential_path("STOP")
    assert is_credential_path("accounts/account-01.json")
    assert is_credential_path("configs/secrets.yaml")
    assert not is_credential_path("agentics/graph/nodes.py")
    assert not is_credential_path("experiments/bench.py")


# ------------------------------------------------------------ layer 1: write gate
def test_non_exempt_task_cannot_even_write_lane_files(mem, tmp_path):
    ws = FakeWS(tmp_path, [])
    ctx = _impl_ctx(mem, ws, exempt=False)
    ws.guard = ctx.local.guard
    plan = {"files": [{"path": "agentics/evolution/scoring.py", "change": "new"}]}
    state = _impl_state(exempt=False, files=plan["files"])
    state["plan"] = json.dumps(plan)
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert ws.refused == ["agentics/evolution/scoring.py"]
    assert not ws.pushed and "P10 protected path" in out["yield_reason"]


def test_exempt_task_writes_agent_code(mem, tmp_path):
    """The dead-gate fix: an owner-exempt task can now write agentics/ files."""
    ws = FakeWS(tmp_path, ["agentics/evolution/scoring.py"])
    ctx = _impl_ctx(mem, ws, exempt=True)
    ws.guard = ctx.local.guard
    plan = {"files": [{"path": "agentics/evolution/scoring.py", "change": "new"}]}
    state = _impl_state(exempt=True, files=plan["files"])
    state["plan"] = json.dumps(plan)
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert ws.written == ["agentics/evolution/scoring.py"]
    assert ws.pushed and out["action"] == "open_pr"


def test_exempt_task_still_bound_by_evolution_allowlist(mem, tmp_path):
    """The exemption opens the write gate, but the pre-existing evolution
    allowlist (node_implement) still confines exempt work to
    EVOLUTION_ALLOWED_PATHS — agentics/graph/nodes.py is yielded there."""
    ws = FakeWS(tmp_path, ["agentics/graph/nodes.py"])
    ctx = _impl_ctx(mem, ws, exempt=True)
    ws.guard = ctx.local.guard
    plan = {"files": [{"path": "agentics/graph/nodes.py", "change": "edit"}]}
    state = _impl_state(exempt=True, files=plan["files"])
    state["plan"] = json.dumps(plan)
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert not ws.pushed and "outside allowed paths" in out["yield_reason"]


def test_exempt_task_never_writes_credentials(mem, tmp_path):
    ws = FakeWS(tmp_path, [])
    ctx = _impl_ctx(mem, ws, exempt=True)
    ws.guard = ctx.local.guard
    plan = {"files": [{"path": ".env", "change": "edit"}]}
    state = _impl_state(exempt=True, files=plan["files"])
    state["plan"] = json.dumps(plan)
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert ws.refused == [".env"]
    assert not ws.pushed


# ------------------------------------------------------------ layer 2: diff gate
def test_diff_gate_exempt_passes_agent_code_but_never_credentials(mem):
    guard = ProtectedPathsGuard(writable_roots=WRITABLE_ROOTS)
    tid = 1
    mem.create_task("self-evolution: prompts", category="evolution")
    mem.set_task_exempt(tid, True, agent="owner")
    # agent code + open-zone file → clean for the exempt task
    assert guard.check_task(mem, tid, [
        "agentics/graph/nodes.py", "tests/test_roles.py"]) == []
    # credentials stay forbidden (the .env / STOP leak fix)
    assert guard.check_task(mem, tid, [".env"]) == [".env"]
    assert guard.check_task(mem, tid, ["STOP"]) == ["STOP"]
    assert guard.check_task(mem, tid, ["accounts/a.json"]) == ["accounts/a.json"]


def test_diff_gate_without_exemption_blocks_everything_protected(mem):
    guard = ProtectedPathsGuard(writable_roots=WRITABLE_ROOTS)
    assert guard.check_task(mem, None, ["agentics/evolution/scoring.py"]) != []


# ------------------------------------------------------------ layers 3-4
def test_merge_gate_refuses_credential_diff_even_when_exempt(mem):
    """The leak fix: an exempt task's MR carrying .env can never merge."""
    from agentics.local_ops import LocalMergeQueue

    guard = ProtectedPathsGuard(writable_roots=WRITABLE_ROOTS)
    task_id = mem.create_task("self-evolution: prompts", category="evolution")
    mem.set_task_exempt(task_id, True, agent="owner")
    mr_id = mem.open_review(
        task_id, "[a04] evolution", "b", author="a04",
        branch=f"agent/a04/task-{task_id}-x",
        files=[".env"])                              # credentials: never
    mem.mq_enqueue(mr_id, author="a04")
    queue = LocalMergeQueue(mem, auto_merge=True, guard=guard)
    queue.drain("a09", verify=lambda mr: (True, ""), merge=lambda mr: True)
    assert mem.get_review(mr_id)["status"] == "failed"
    assert mem.get_task(task_id)["status"] == "failed"


def test_merge_gate_passes_agent_code_of_exempt_task(mem):
    from agentics.local_ops import LocalMergeQueue

    guard = ProtectedPathsGuard(writable_roots=WRITABLE_ROOTS)
    task_id = mem.create_task("self-evolution: prompts", category="evolution")
    mem.set_task_exempt(task_id, True, agent="owner")
    mr_id = mem.open_review(
        task_id, "[a04] evolution", "b", author="a04",
        branch=f"agent/a04/task-{task_id}-x",
        files=["agentics/evolution/scoring.py"])    # sanctioned agent code
    mem.mq_enqueue(mr_id, author="a04")
    queue = LocalMergeQueue(mem, auto_merge=True, guard=guard)
    merged = queue.drain("a09", verify=lambda mr: (True, ""), merge=lambda mr: True)
    assert merged == 1
    assert mem.get_review(mr_id)["status"] == "merged"
