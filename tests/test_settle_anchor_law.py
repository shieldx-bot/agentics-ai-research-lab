"""The settle anchor law + the discovery double-judge gate.

The user's epistemic axis, turned into code and tests: NOT "algorithm vs
AI" (an algorithmic scorer can be just as self-graded), but — *who holds
the pen that writes the rubric, and does the rubric anchor to something
outside the agent's reach?* The one-question audit for every verdict:

    "Among the inputs that DECIDE this verdict, does at least ONE exist
     that the agent did not author?"

Covered here:

- VerdictProvenance: self_authored_ratio math (bare machine verdict =
  1.0; one external anchor = 0.75) and the settle law (supported requires
  ratio < 1; anchorless supported -> `unanchored`; refuted passes).
- detect_pinned_data: the protocol declares `# AGENTICS_ANCHOR:
  pinned_data <path> <sha256>`; the anchor holds only when the workspace
  file hashes to the declaration.
- Ledger provenance: every settle records self_authored_ratio + anchors
  NEXT TO the verdict.
- Memory: owner_approve (the local-only escape hatch) + the P24 selector
  prioritizing unanchored claims (replication is their path to support).
- settle_hypothesis + node_verify: anchorless supported downgraded +
  journaled `settle_unanchored`; citation-anchored supported recorded.
- Replication: an independent peer's confirmed re-run PROMOTES an
  unanchored claim to supported (replication_promoted).
- Discovery double-judge gate: mark_discovery only when a SECOND account
  (view() of an independent identity) agrees AND its answer carries an
  external citation URL; disagreement / missing citations / single-account
  pools deny the gate and the loop keeps the plain follow-up behavior.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.verify.anchors import (  # noqa: E402
    ANCHOR_KINDS, AUTHORED_INPUTS, VerdictProvenance, apply_settle_law,
    detect_pinned_data, provenance_for_settle,
)
from agentics.verify.experiment import (  # noqa: E402
    Hypothesis, register_hypothesis, run_hypothesis, settle_hypothesis,
)
from agentics.verify_metrics import Criterion  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.research import ideation as idea  # noqa: E402


# ------------------------------------------------------------ pure provenance

def test_ratio_bare_machine_verdict_is_fully_self_authored():
    prov = provenance_for_settle()
    assert prov.ratio == 1.0 and prov.authored == list(AUTHORED_INPUTS)
    assert not prov.anchored
    # the law: anchorless supported is downgraded
    assert apply_settle_law("supported", prov) == ("unanchored", True)
    # the law guards OPTIMISM, not pessimism
    assert apply_settle_law("refuted", prov) == ("refuted", False)
    assert apply_settle_law("inconclusive", prov) == ("inconclusive", False)
    assert apply_settle_law("supported", prov, require_anchor=False) == \
        ("supported", False)


def test_ratio_one_external_anchor_drops_below_one():
    for anchor_kwargs in ("citations", "peer_judge", "replicated",
                          "owner_approved", "pinned_data", "ci_verdict"):
        kwargs = ({anchor_kwargs: 3} if anchor_kwargs == "citations"
                  else {anchor_kwargs: True})
        prov = provenance_for_settle(**kwargs)
        assert prov.anchored and prov.ratio < 1.0
        assert apply_settle_law("supported", prov) == ("supported", False)
    # citation is a COUNT (Perplexity returns a list of URLs)
    assert provenance_for_settle(citations=3).anchors == ["citation"]
    assert provenance_for_settle(citations=0).anchors == []
    # anchors dedupe and stay inside the known vocabulary
    prov = VerdictProvenance()
    prov.add_anchor("citation", "citation", "vibes", "replication")
    assert prov.anchors == ["citation", "replication"]


def test_detect_pinned_data_requires_matching_sha256(tmp_path):
    class WS:
        def __init__(self, root: Path) -> None:
            self.root = root

        def read_file(self, rel: str) -> str:
            return (self.root / rel).read_text(encoding="utf-8")

    ws = WS(tmp_path)
    (tmp_path / "data.csv").write_text("x,y\n1,2\n3,4\n")
    good = hashlib.sha256(b"x,y\n1,2\n3,4\n").hexdigest()
    anchored = (f"# AGENTICS_ANCHOR: pinned_data data.csv {good}\n"
                "print('AGENTICS_RESULT: {\"m\": 1}')")
    assert detect_pinned_data(ws, anchored) is True
    forged = anchored.replace(good, "0" * 64)
    assert detect_pinned_data(ws, forged) is False          # hash mismatch
    assert detect_pinned_data(ws, "# AGENTICS_ANCHOR: pinned_data "
                                  "missing.csv " + good) is False  # no file
    assert detect_pinned_data(ws, "print('no declaration')") is False


# ------------------------------------------------------------------- memory

@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _hyp(mem, claim, status="proposed", agent="a02"):
    return mem.add_hypothesis(
        agent, claim, {"metric": "m", "op": ">=", "target": 1}, status=status)


def test_ledger_row_records_provenance_next_to_verdict(mem):
    hid = _hyp(mem, "a claim whose verdict provenance is recorded")
    mem.record_experiment("a02", hid, "supported", metric="m", measured=2.0,
                          op=">=", target=1.0, self_authored_ratio=0.75,
                          anchors='["citation"]')
    row = mem.ledger_recent(1)[0]
    assert row["status"] == "supported"
    assert row["self_authored_ratio"] == pytest.approx(0.75)
    assert json.loads(row["anchors"]) == ["citation"]


def test_owner_approve_is_a_verdict_input_outside_the_agent_reach(mem):
    hid = _hyp(mem, "a claim the owner explicitly accepts")
    assert mem.settle_anchor_state(hid) == {"replicated": False,
                                            "owner_approved": False}
    assert mem.owner_approve(hid, note="I re-derived this myself") is True
    assert mem.settle_anchor_state(hid)["owner_approved"] is True
    # idempotent + audited
    assert mem.owner_approve(hid) is False
    events = [e["event"] for e in mem.recent_journal(limit=10)]
    assert "owner_approved" in events


def test_replication_selector_prioritizes_unanchored(mem):
    old_supported = _hyp(mem, "an old supported claim worth re-running",
                         status="supported", agent="a01")
    mem.update_hypothesis(old_supported, "supported")
    unanchored = _hyp(mem, "a fresh unanchored claim awaiting replication",
                      status="unanchored", agent="a01")
    for h in (old_supported, unanchored):
        mem.update_hypothesis_protocol_code(
            h, f"print('AGENTICS_RESULT: {{\"m\": 1}}')  # h{h}")
    got = mem.hypotheses_needing_replication("a09", limit=2)
    assert [h["id"] for h in got] == [unanchored, old_supported]


# ------------------------------------------------------- settle_hypothesis

def _hypothesis(code: str, target: float = 0.9) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="anchor-law-check", claim="the protocol measures the "
        "pre-registered metric above the target",
        prediction="AGENTICS_RESULT prints the metric", 
        criterion=Criterion.from_dict({"metric": "precision", "op": ">=",
                                       "target": target}),
        code=code, task=1,
    )


class WS:
    """Minimal workspace for run_hypothesis: writes scripts and executes
    them through the real run_protocol path (subprocess python)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def sync_main(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_file(self, rel: str, content: str) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        script = self.root / cmd[1]
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=timeout, cwd=self.root,
                              check=False)
        return proc.returncode, proc.stdout + proc.stderr


PASSING = "print('AGENTICS_RESULT: {\"precision\": 0.97}')"


def test_settle_law_downgrades_anchorless_supported(mem, tmp_path):
    ws = WS(tmp_path)
    h = _hypothesis(PASSING)
    hyp_id = register_hypothesis(mem, "a02", h)
    ev = run_hypothesis(ws, h)
    ev_id = settle_hypothesis(mem, "a02", hyp_id, h, ev)   # no anchor passed
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "unanchored"                   # supported WITHHELD
    assert "unanchored" in row["verdict_note"]
    ledger = mem.ledger_recent(1)[0]
    assert ledger["status"] == "unanchored"
    assert ledger["self_authored_ratio"] == pytest.approx(1.0)
    assert json.loads(ledger["anchors"]) == []
    events = {e["event"] for e in mem.recent_journal(limit=10, agent="a02")}
    assert "settle_unanchored" in events
    # an unanchored claim is NOT common ground yet
    assert mem.verified_knowledge() == []


def test_settle_law_keeps_supported_when_a_citation_anchors_it(mem, tmp_path):
    ws = WS(tmp_path)
    h = _hypothesis(PASSING)
    hyp_id = register_hypothesis(mem, "a02", h)
    ev = run_hypothesis(ws, h)
    settle_hypothesis(mem, "a02", hyp_id, h, ev, citations=2)
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "supported"
    ledger = mem.ledger_recent(1)[0]
    assert ledger["self_authored_ratio"] == pytest.approx(0.75)
    assert json.loads(ledger["anchors"]) == ["citation"]
    assert [k["status"] for k in mem.verified_knowledge()] == ["supported"]


def test_settle_law_accepts_owner_approval_and_pinned_data(mem, tmp_path):
    ws = WS(tmp_path)
    # owner approval read from the hypothesis row
    h = _hypothesis(PASSING)
    hyp_id = register_hypothesis(mem, "a02", h)
    ev = run_hypothesis(ws, h)
    mem.owner_approve(hyp_id, note="reviewed the protocol myself")
    settle_hypothesis(mem, "a02", hyp_id, h, ev)
    assert mem.get_hypothesis(hyp_id)["status"] == "supported"
    assert json.loads(mem.ledger_recent(1)[0]["anchors"]) == ["owner_approval"]
    # pinned-data detected INSIDE run_hypothesis from the protocol source
    data = "rows,metric\n100,0.9\n"
    digest = hashlib.sha256(data.encode()).hexdigest()
    rel = "research_notes/experiments/dataset.csv"
    h2 = _hypothesis(
        f"open('{rel}').read()\n{PASSING}\n"
        f"# AGENTICS_ANCHOR: pinned_data {rel} {digest}\n")
    hid2 = register_hypothesis(mem, "a02", h2)
    (tmp_path / "research_notes" / "experiments").mkdir(parents=True,
                                                        exist_ok=True)
    (tmp_path / rel).write_text(data)
    ev2 = run_hypothesis(ws, h2)
    assert ev2.meta.get("anchor_pinned_data") is True
    settle_hypothesis(mem, "a02", hid2, h2, ev2)
    assert mem.get_hypothesis(hid2)["status"] == "supported"
    assert json.loads(mem.ledger_recent(1)[0]["anchors"]) == ["pinned_data"]


# --------------------------------------------------------------- node paths

PROTOCOL = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'effect_size': 0.3}))\n"
)


class FakeLLM:
    account = SimpleNamespace(name="offline")

    def __init__(self, *answers: tuple[str, str], citations=()) -> None:
        self.answers = list(answers)
        self.citations = list(citations)
        self.calls: list[tuple[str, str]] = []

    async def ask(self, prompt, *, mode="concise", system=None,
                  purpose="general"):
        self.calls.append((purpose, prompt))
        for i, (marker, payload) in enumerate(self.answers):
            if marker in prompt:
                del self.answers[i]
                return SimpleNamespace(answer=payload,
                                       citations=list(self.citations), ok=True)
        return SimpleNamespace(answer="", citations=[], ok=False)


class FakeWS:
    def __init__(self, root: Path) -> None:
        import subprocess

        self.path = root
        self._subprocess = subprocess

    def sync_main(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        script = self.path / cmd[1]
        proc = self._subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            timeout=timeout, cwd=self.path, check=False)
        return proc.returncode, proc.stdout + proc.stderr


class LawSettings:
    research_topic = "NFV"
    diversity_jaccard_max = 0.8
    ideation_enabled = True
    idea_min_score = 60
    idea_repair_margin = 20
    idea_max_children = 2
    idea_max_attempts = 2
    world_context_notes = 12
    verify_protocol_repair_rounds = 0
    discovery_novelty_min = 70
    branch_max_depth = 4
    settle_require_anchor = True      # THE LAW IS ON
    discovery_second_judge = True


def _node_ctx(mem, tmp_path, llm, settings=None):
    from agentics.agents.roles import Role

    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, memory=mem, llm=llm,
        workspace=FakeWS(tmp_path / "ws"), cycle=1,
        settings=settings or LawSettings(),
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"_system_prompt": "sys"},
    )


PROTOCOL_REPLY = json.dumps({"hypotheses": [{
    "claim": "the toy protocol effect size clears the pre-set bar",
    "metric": "effect_size", "op": ">=", "target": 0.2,
    "protocol": PROTOCOL}]})


def test_node_verify_downgrades_anchorless_supported(tmp_path):
    from agentics.graph import nodes

    mem = SharedMemory(tmp_path / "m.db")
    llm = FakeLLM(("Extract up to 2", PROTOCOL_REPLY),
                  ("ROLE: ideator", json.dumps({"hypotheses": []})))
    ctx = _node_ctx(mem, tmp_path, llm)
    state = {"finding_text": "Finding: the toy protocol measures 0.3",
             "citations": [], "done": False}
    asyncio.run(nodes.node_verify(ctx, state))
    rows = mem.list_hypotheses(limit=5)
    assert rows and rows[0]["status"] == "unanchored"
    ledger = mem.ledger_recent(1)[0]
    assert ledger["status"] == "unanchored"
    assert ledger["self_authored_ratio"] == pytest.approx(1.0)
    events = {e["event"] for e in mem.recent_journal(limit=12, agent="a02")}
    assert "settle_unanchored" in events
    # the world model holds it as a provisional finding, NOT verified
    world = mem.world_render(limit=10)
    assert "-> unanchored" in world
    assert "[VERIFIED]" not in world


def test_node_verify_research_citations_anchor_the_settle(tmp_path):
    from agentics.graph import nodes

    mem = SharedMemory(tmp_path / "m.db")
    llm = FakeLLM(("Extract up to 2", PROTOCOL_REPLY),
                  ("ROLE: ideator", json.dumps({"hypotheses": []})))
    ctx = _node_ctx(mem, tmp_path, llm)
    state = {"finding_text": "Finding: the toy protocol measures 0.3",
             "citations": ["https://arxiv.org/abs/2511.02824"],
             "done": False}
    asyncio.run(nodes.node_verify(ctx, state))
    rows = mem.list_hypotheses(limit=5)
    assert rows and rows[0]["status"] == "supported"      # anchored: stays
    ledger = mem.ledger_recent(1)[0]
    assert json.loads(ledger["anchors"]) == ["citation"]
    assert ledger["self_authored_ratio"] == pytest.approx(0.75)
    assert [k["status"] for k in mem.verified_knowledge()] == ["supported"]


# ------------------------------------------------------- replication promote

def test_independent_replication_promotes_unanchored(mem, tmp_path):
    from agentics.verify.replication import replicate_candidate

    hid = _hyp(mem, "a measured claim waiting for an independent peer",
               status="unanchored", agent="a01")
    mem.update_hypothesis_protocol_code(hid, PROTOCOL)
    mem.update_hypothesis(hid, "unanchored")
    # the P24 provenance row needs the author's evidence row to attach to
    author_ev = mem.record_evidence("a01", None, "protocol", 0, "confirmed",
                                    hypothesis="waiting", method="protocol")

    class PeerWS:
        def __init__(self, root: Path) -> None:
            self.root = root

        def write_file(self, rel, content):
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        def run_command(self, cmd, timeout=120):
            script = self.root / cmd[1]
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True,
                                  timeout=timeout, cwd=self.root, check=False)
            return proc.returncode, proc.stdout + proc.stderr

    report = replicate_candidate(mem, PeerWS(tmp_path / "peer"),
                                 {"id": hid, "agent": "a01",
                                  "evidence_id": author_ev,
                                  "criterion": {"metric": "effect_size",
                                                "op": ">=", "target": 0.2},
                                  "protocol_code": PROTOCOL,
                                  "claim": "waiting", "status": "unanchored"},
                                 agent="a09")
    assert report["replicated"] is True
    assert report["event"] == "replication_promoted"
    assert mem.get_hypothesis(hid)["status"] == "supported"
    promote = [r for r in mem.ledger_recent(5) if r["source"] == "replication"]
    assert promote and promote[0]["hyp_id"] == hid
    assert promote[0]["self_authored_ratio"] == pytest.approx(0.75)
    assert json.loads(promote[0]["anchors"]) == ["replication"]
    events = {e["event"] for e in mem.recent_journal(limit=10, agent="a09")}
    assert "replication_promoted" in events


# ------------------------------------------------- discovery double-judge gate

def test_parse_novelty_judgement_contract():
    ok = json.dumps({"novelty": 85, "is_discovery": True, "reason": "surprise"})
    assert idea.parse_novelty_judgement(ok) == (85.0, True, "surprise")
    assert idea.parse_novelty_judgement("prose") == (None, False, "")
    assert idea.parse_novelty_judgement(
        '{"novelty": "high", "is_discovery": true}') == (None, False, "")


class FakePeer:
    """One independent account: the second judge's queued answer."""

    def __init__(self, answer: str, citations: list[str]) -> None:
        self.answer = answer
        self.citations = citations
        self.prompts: list[str] = []

    async def ask(self, prompt, *, mode="concise", system=None,
                  purpose="general"):
        self.prompts.append(prompt)
        return SimpleNamespace(answer=self.answer, citations=self.citations)


class GateHost:
    """Session-host fake: the main account answers through FakeLLM; view(i)
    hands out the pre-queued PEER accounts (the independent second judge).
    `pool_size` simulates how many identities the deployment has."""

    def __init__(self, main: FakeLLM, peers: list[FakePeer],
                 pool_size: int = 2) -> None:
        self.main = main
        self.peers = list(peers)
        self.used_peers: list[FakePeer] = []
        self.pool_size = pool_size
        self.prefer_index = 0

    def __len__(self) -> int:
        return self.pool_size

    @property
    def account(self):
        return self.main.account

    async def ask(self, prompt, *, mode="concise", system=None,
                  purpose="general"):
        return await self.main.ask(prompt, mode=mode, system=system,
                                   purpose=purpose)

    def view(self, idx: int) -> FakePeer:
        peer = self.peers.pop(0) if self.peers else FakePeer("", [])
        self.used_peers.append(peer)
        return peer


def _gate_ctx(mem, tmp_path, host):
    ctx = _node_ctx(mem, tmp_path, host)
    return ctx


def _discovery_reply(novelty=92):
    return json.dumps({
        "novelty": novelty, "is_discovery": True,
        "reason": "an unexplained effect the world model cannot account for",
        "hypotheses": [
            {"claim": "the discovery effect is caused by the retry backoff "
                      "interaction rather than the scheduler",
             "metric": "effect_size", "op": ">=", "target": 0.2,
             "angle": "mechanism"},
        ]})


GOOD_JUDGE = json.dumps({"non_obviousness": 85, "clarity": 80,
                         "feasibility": 70, "soundness": 8,
                         "notes": "solid mechanism, clean test path"})

PEER_AGREES = json.dumps({"novelty": 88, "is_discovery": True,
                          "reason": "independent check agrees it is a surprise"})
PEER_DENIES = json.dumps({"novelty": 40, "is_discovery": False,
                          "reason": "routine confirmation of a known result"})


def test_harvest_marks_discovery_only_when_peer_agrees_with_citation(
        mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim about autoscaling policies",
                  status="supported")
    host = GateHost(
        FakeLLM(("ROLE: ideator", _discovery_reply()),
                ("ROLE: judge", GOOD_JUDGE)),
        peers=[FakePeer(PEER_AGREES, ["https://example.com/paper"])])
    ctx = _gate_ctx(mem, tmp_path, host)
    n = asyncio.run(_harvest_insights(ctx, parent, "parent claim",
                                      "supported", "speedup=2.4"))
    assert n == 1
    row = mem.get_hypothesis(parent)
    assert row["discovery"] is True and row["novelty"] == pytest.approx(92.0)
    # the INDEPENDENT account was actually consulted (peer consumed once)
    assert len(host.used_peers) == 1
    peer_prompt = host.used_peers[0].prompts[0]
    assert "independent novelty judge" in peer_prompt
    assert "is_discovery" in peer_prompt
    events = {e["event"] for e in mem.recent_journal(limit=12, agent="a02")}
    assert "discovery_detected" in events


def test_harvest_gate_denied_when_second_judge_disagrees(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim for denial", status="supported")
    host = GateHost(
        FakeLLM(("ROLE: ideator", _discovery_reply()),
                ("ROLE: judge", GOOD_JUDGE)),
        peers=[FakePeer(PEER_DENIES, ["https://example.com/x"])])
    ctx = _gate_ctx(mem, tmp_path, host)
    n = asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert n == 1                              # plain follow-up still registers
    assert mem.get_hypothesis(parent)["discovery"] is False
    kid = [r for r in mem.list_hypotheses(limit=5)
           if r["parents"] == [parent]][0]
    assert kid["branch"] == ""                 # not an angled branch
    events = {e["event"] for e in mem.recent_journal(limit=12, agent="a02")}
    assert "discovery_gate_denied" in events
    assert "discovery_detected" not in events


def test_harvest_gate_denied_without_external_citation(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim without citation", status="supported")
    host = GateHost(
        FakeLLM(("ROLE: ideator", _discovery_reply()),
                ("ROLE: judge", GOOD_JUDGE)),
        peers=[FakePeer(PEER_AGREES, [])])     # agrees but cites NOTHING
    ctx = _gate_ctx(mem, tmp_path, host)
    asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert mem.get_hypothesis(parent)["discovery"] is False
    events = {e["event"] for e in mem.recent_journal(limit=12, agent="a02")}
    assert "discovery_gate_denied" in events
    assert any("citation" in (e.get("detail") or "") for e in
               mem.recent_journal(limit=12, agent="a02"))


def test_harvest_gate_denied_in_a_single_account_pool(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim in a lonely pool", status="supported")
    host = GateHost(FakeLLM(("ROLE: ideator", _discovery_reply()),
                            ("ROLE: judge", GOOD_JUDGE)), peers=[],
                    pool_size=1)               # one identity: no peer exists
    ctx = _gate_ctx(mem, tmp_path, host)
    asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    assert mem.get_hypothesis(parent)["discovery"] is False
    denied = [e for e in mem.recent_journal(limit=12, agent="a02")
              if e["event"] == "discovery_gate_denied"]
    assert denied
    assert "pool=1" in json.loads(denied[0]["detail"])["why"]


class LawSettingsOff(LawSettings):
    discovery_second_judge = False             # config off = legacy behavior


def test_harvest_gate_respects_config_off_switch(mem, tmp_path):
    from agentics.graph.nodes import _harvest_insights

    parent = _hyp(mem, "settled parent claim with gate disabled", status="supported")
    host = GateHost(FakeLLM(("ROLE: ideator", _discovery_reply()),
                            ("ROLE: judge", GOOD_JUDGE)), peers=[],
                    pool_size=1)               # no peer available anyway
    ctx = _gate_ctx(mem, tmp_path, host)
    ctx.settings = LawSettingsOff()
    asyncio.run(_harvest_insights(ctx, parent, "claim", "supported", "x=1"))
    # legacy behavior: the FIRST judge alone decides, no peer consulted
    assert mem.get_hypothesis(parent)["discovery"] is True
    assert host.used_peers == []
