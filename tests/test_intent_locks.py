"""P13 file locks + P14 intent binding (scope enforcement)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard
from agentics.policies.intent import (
    FileLockManager, path_covered, scope_violations,
)


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _locks(mem) -> FileLockManager:
    return FileLockManager(mem, ttl_s=1800)


# ------------------------------------------------------------ primitives
def test_path_covered_exact_prefix_and_glob():
    assert path_covered("experiments/e.py", ["experiments/e.py"])
    assert path_covered("experiments/a/b.py", ["experiments/"])
    assert path_covered("experiments/e.py", ["experiments/*.py"])
    assert not path_covered("agentics/x.py", ["experiments/"])
    assert not path_covered("experiments/e.py", [])


def test_scope_violations_lists_undeclared_files():
    drift = scope_violations(["experiments/ok.py", "docs/undeclared.md"],
                             ["experiments/"])
    assert drift == ["docs/undeclared.md"]
    assert scope_violations(["experiments/ok.py"], ["experiments/"]) == []


# ------------------------------------------------------------ FileLockManager
def test_try_acquire_all_is_all_or_nothing(mem):
    locks = _locks(mem)
    assert locks.try_acquire_all("a07", ["experiments/e.py"])[0]
    ok, conflicts = locks.try_acquire_all("a04", ["experiments/free.py",
                                                  "experiments/e.py"])
    assert not ok and "experiments/e.py" in conflicts
    assert conflicts["experiments/e.py"] == "a07"
    # rollback: a04 must NOT half-hold the free file
    assert all(l["agent"] != "a04" for l in locks.active())


def test_lock_ttl_expires_and_sweeps(mem):
    locks = FileLockManager(mem, ttl_s=0.05)
    locks.try_acquire_all("a07", ["experiments/e.py"])
    import time
    time.sleep(0.08)
    assert locks.active() == []
    assert locks.sweep_stale() == ["experiments/e.py"]
    assert locks.try_acquire_all("a04", ["experiments/e.py"])[0]


def test_release_agent_frees_everything(mem):
    locks = _locks(mem)
    locks.try_acquire_all("a04", ["experiments/a.py", "docs/b.md"])
    assert locks.release_agent("a04") == ["docs/b.md", "experiments/a.py"]
    assert locks.active() == []


# ------------------------------------------------------------ node_plan hook
def _ctx(mem, agent_id="a04", local=None):
    return SimpleNamespace(
        agent_id=agent_id, role=SimpleNamespace(value="coder"),
        memory=mem, cycle=1, llm=None,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=0),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=local,
        workspace=SimpleNamespace(sync_main=lambda: None),
        claims=SimpleNamespace(release=lambda *a, **k: None),
    )


PLAN = {
    "files": [{"path": "experiments/bench.py", "change": "new"}],
    "steps": ["write bench"],
    "commit_message": "add bench",
    "reuses": [],
    "reusable_tools": [],
}


def _patch_llm(monkeypatch, payload):
    async def fake(ctx, prompt, mode="concise", purpose="general", state=None):
        return json.dumps(payload), []
    monkeypatch.setattr(nodes, "_llm_ask", fake)


def _plan_state():
    return {
        "issue_number": 3,
        "issue_title": "Experiment: bench",
        "issue_body": "",
        "papers": [{"answer": ""}],
        "category": "experiment",
    }


def test_plan_acquires_locks_for_declared_files(mem, monkeypatch):
    locks = _locks(mem)
    local = SimpleNamespace(guard=ProtectedPathsGuard(), locks=locks)
    ctx = _ctx(mem, local=local)
    _patch_llm(monkeypatch, PLAN)
    state = asyncio.run(nodes.node_plan(ctx, _plan_state()))
    assert state["expected_files"] == ["experiments/bench.py"]
    holders = {l["agent"] for l in locks.active()}
    assert holders == {"a04"}
    events = [e["event"] for e in mem.recent_journal(limit=5)]
    assert "intent_bound" in events


def test_plan_yields_when_another_agent_holds_the_file(mem, monkeypatch):
    locks = _locks(mem)
    locks.try_acquire_all("a07", ["experiments/bench.py"])
    local = SimpleNamespace(guard=ProtectedPathsGuard(), locks=locks)
    ctx = _ctx(mem, local=local)
    _patch_llm(monkeypatch, PLAN)
    state = asyncio.run(nodes.node_plan(ctx, _plan_state()))
    assert state["done"] and "P13" in state["yield_reason"]
    assert any(e["event"] == "lock_conflict" for e in mem.recent_journal(limit=5))
    # a04 must not hold anything (all-or-nothing even on the yield path)
    assert all(l["agent"] != "a04" for l in locks.active())


def test_plan_without_lock_manager_still_binds_scope(mem, monkeypatch):
    local = SimpleNamespace(guard=ProtectedPathsGuard())      # no locks (legacy ctx)
    ctx = _ctx(mem, local=local)
    _patch_llm(monkeypatch, PLAN)
    state = asyncio.run(nodes.node_plan(ctx, _plan_state()))
    assert state["expected_files"] == ["experiments/bench.py"]
    assert "yield_reason" not in state


# ------------------------------------------------------------ node_implement hook
class FakeWS:
    def __init__(self, changed):
        self.path = Path(".")
        self._changed = changed

    def sync_main(self):
        pass

    def checkout_new_branch(self, branch):
        pass

    def read_file(self, path):
        return ""

    def write_file(self, path, content):
        pass

    def changed_files(self):
        return list(self._changed)

    def commit_and_push(self, branch, message, allow_protected=False):
        return True


def _impl_ctx(mem, ws, agent_id="a04", local=None):
    ctx = _ctx(mem, agent_id=agent_id, local=local)
    ctx.workspace = ws
    from agentics.policies.ownership import OwnershipPolicy
    ctx.ownership = OwnershipPolicy()
    return ctx


def test_implement_yields_on_scope_drift(mem, monkeypatch):
    ws = FakeWS(["experiments/bench.py", "docs/undeclared.md"])
    local = SimpleNamespace(guard=ProtectedPathsGuard(),
                            locks=_locks(mem))
    ctx = _impl_ctx(mem, ws, local=local)
    state = {
        "branch": "agent/a04/task-3-x", "issue_number": 3,
        "issue_title": "Experiment: bench", "category": "experiment",
        "plan": json.dumps(PLAN), "expected_files": ["experiments/bench.py"],
    }
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert out["done"] and "P14" in out["yield_reason"]
    assert any(e["event"] == "scope_violation"
               for e in mem.recent_journal(limit=5))


def test_implement_passes_when_diff_stays_in_scope(mem, monkeypatch):
    ws = FakeWS(["experiments/bench.py"])
    local = SimpleNamespace(guard=ProtectedPathsGuard(),
                            locks=_locks(mem))
    ctx = _impl_ctx(mem, ws, local=local)
    # a real LLM produces the file content; without one the cycle correctly
    # degrades to state["error"] (resilience ported from main), which would
    # defeat the point of this scope test
    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        return "# bench\n", []
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = {
        "branch": "agent/a04/task-3-x", "issue_number": 3,
        "issue_title": "Experiment: bench", "category": "experiment",
        "plan": json.dumps(PLAN), "expected_files": ["experiments/bench.py"],
    }
    out = asyncio.run(nodes.node_implement(ctx, state))
    assert "yield_reason" not in out
    assert "error" not in out or not out["error"]


# ------------------------------------------------------------ node_reflect hook
def test_reflect_releases_the_agents_locks(mem):
    locks = _locks(mem)
    locks.try_acquire_all("a04", ["experiments/bench.py"])
    local = SimpleNamespace(guard=ProtectedPathsGuard(), locks=locks)
    ctx = _ctx(mem, local=local)
    nodes.node_reflect(ctx, {"summary": "resting"})
    assert locks.active() == []
    assert any(e["event"] == "locks_released"
               for e in mem.recent_journal(limit=5))
