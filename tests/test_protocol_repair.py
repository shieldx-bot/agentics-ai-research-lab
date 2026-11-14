"""P16-R — bounded self-repair for inconclusive protocols (offline tests).

The machine verdict was always correct; the weak spot was that the lab
settled `inconclusive` and discarded the failure output. These tests pin
the repair loop: the failure note reaches the author's prompt, the fixed
protocol re-runs through the SAME machine adjudication, the criterion
cannot move, unchanged/no-op replies do not burn a re-run, and every
round is journalled with before/after verdicts.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.config import Settings  # noqa: E402
from agentics.graph.nodes import _repair_protocol_rounds  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify.repair import (  # noqa: E402
    build_repair_prompt, extract_fixed_code,
)
from agentics.verify_metrics import Criterion, parse_metrics  # noqa: E402
from agentics.verify_ops import Evidence  # noqa: E402


def _memory() -> SharedMemory:
    s = Settings()
    s.data_dir = Path(tempfile.mkdtemp(prefix="p16r-"))
    s.ensure_dirs()
    return SharedMemory(s.db_path)


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ask(self, prompt, mode="concise", purpose="general", system=None):
        self.prompts.append(prompt)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(ok=True, answer=item, citations=[])


def _ctx(memory, llm, rounds=1):
    s = Settings()
    s.verify_protocol_repair_rounds = rounds
    s.adaptive_prompts = False        # the repair ask must not trigger a
    return SimpleNamespace(           # role-mission synthesis in the fake
        agent_id="a01", memory=memory, llm=llm, settings=s, budget=None,
        meta={}, role=None)


_CRIT = Criterion("precision", ">=", 0.9)
_BROKEN = "print('measuring...')\n# forgot the AGENTICS_RESULT line"
_FIXED = ("data = [0.95, 0.97]\n"
          'print("AGENTICS_RESULT:", {"precision": sum(data) / len(data)})')


def _inconclusive():
    return Evidence(cmd="experiments/h1.py", exit_code=0, verdict="inconclusive",
                    method="protocol", output="measuring...",
                    meta={"note": "run 1 printed no parsable AGENTICS_RESULT line",
                          "metrics": {}, "runs": []})


# ------------------------------------------------------------- pure helpers
def test_repair_prompt_carries_contract_failure_and_frozen_criterion():
    prompt = build_repair_prompt("RF reaches >= 0.9 precision", _CRIT,
                                 _BROKEN, _inconclusive())
    assert "AGENTICS_RESULT" in prompt              # the contract
    assert "precision >= 0.9" in prompt             # the frozen criterion
    assert "no parsable AGENTICS_RESULT" in prompt  # the machine failure note
    assert _BROKEN in prompt                        # the author's own script


def test_extract_fixed_code_strips_fences_and_rejects_noops():
    assert extract_fixed_code(f"```python\n{_FIXED}\n```", _BROKEN) == _FIXED
    assert extract_fixed_code(f"```\n{_FIXED}\n```", _BROKEN) == _FIXED
    assert extract_fixed_code(_FIXED, _BROKEN) == _FIXED
    assert extract_fixed_code(_BROKEN, _BROKEN) is None     # unchanged no-op
    assert extract_fixed_code("", _BROKEN) is None
    assert extract_fixed_code("```python\n   \n```", _BROKEN) is None


def test_parse_metrics_contract_unchanged():
    parsed = parse_metrics("noise\nAGENTICS_RESULT: {\"precision\": 0.93}")
    assert parsed == {"precision": 0.93}


# ------------------------------------------------------------- repair rounds
def test_repair_round_fixes_protocol_and_journals():
    mem = _memory()
    llm = FakeLLM([f"```python\n{_FIXED}\n```"])
    calls = []

    def rerun(code):
        calls.append(code)
        return Evidence(cmd="experiments/h1.py", exit_code=0,
                        verdict="confirmed", method="protocol",
                        output="AGENTICS_RESULT: {\"precision\": 0.96}",
                        meta={"note": "measured precision=0.96 vs precision >= 0.9",
                              "metrics": {"precision": 0.96}, "runs": []})

    ev, final = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm), {}, claim="RF reaches >= 0.9 precision",
        criterion=_CRIT, code=_BROKEN, ev=_inconclusive(),
        rerun=rerun, what="verify:h1", hyp_id=None))
    assert ev.verdict == "confirmed" and final == _FIXED
    assert calls == [_FIXED]                        # exactly one re-run
    assert "no parsable" in llm.prompts[0]          # failure note was fed back
    journal = [j for j in mem.recent_journal(10)
               if j["event"] == "p16_protocol_repair"]
    detail = json.loads(journal[0]["detail"])
    assert detail["verdict_before"] == "inconclusive"
    assert detail["verdict_after"] == "confirmed"


def test_repair_skipped_when_verdict_not_inconclusive():
    mem = _memory()
    llm = FakeLLM([])

    def rerun(code):
        raise AssertionError("must not re-run a decided protocol")

    ev, _ = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm), {}, claim="c", criterion=_CRIT, code=_BROKEN,
        ev=Evidence(cmd="x", exit_code=0, verdict="confirmed", method="protocol",
                    output="", meta={}),
        rerun=rerun, what="verify:h1"))
    assert ev.verdict == "confirmed" and llm.prompts == []


def test_repair_knob_zero_disables():
    mem = _memory()
    llm = FakeLLM([])
    ev, code = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm, rounds=0), {}, claim="c", criterion=_CRIT,
        code=_BROKEN, ev=_inconclusive(),
        rerun=lambda c: (_ for _ in ()).throw(AssertionError("no rerun")),
        what="verify:h1"))
    assert ev.verdict == "inconclusive" and code == _BROKEN
    assert llm.prompts == []


def test_noop_reply_journals_no_usable_fix_and_never_reruns():
    mem = _memory()
    llm = FakeLLM([_BROKEN])                        # author re-sent the same file

    def rerun(code):
        raise AssertionError("a no-op fix must not burn a re-run")

    ev, _ = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm), {}, claim="c", criterion=_CRIT, code=_BROKEN,
        ev=_inconclusive(), rerun=rerun, what="verify:h1"))
    assert ev.verdict == "inconclusive"
    journal = [j for j in mem.recent_journal(10)
               if j["event"] == "p16_protocol_repair"]
    assert json.loads(journal[0]["detail"])["outcome"] == "no_usable_fix"


def test_still_broken_after_round_settles_inconclusive_with_trail():
    mem = _memory()
    llm = FakeLLM([f"```python\n{_BROKEN} x = 1\n```"])   # different but broken

    def rerun(code):
        return Evidence(cmd="experiments/h1.py", exit_code=0,
                        verdict="inconclusive", method="protocol", output="...",
                        meta={"note": "still no parsable line",
                              "metrics": {}, "runs": []})

    ev, final = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm), {}, claim="c", criterion=_CRIT, code=_BROKEN,
        ev=_inconclusive(), rerun=rerun, what="verify:h1"))
    assert ev.verdict == "inconclusive" and final != _BROKEN
    journal = [j for j in mem.recent_journal(10)
               if j["event"] == "p16_protocol_repair"]
    assert json.loads(journal[0]["detail"])["verdict_after"] == "inconclusive"


def test_no_llm_degrades_without_touching_the_verdict():
    mem = _memory()
    ev, code = asyncio.run(_repair_protocol_rounds(
        _ctx(mem, None), {}, claim="c", criterion=_CRIT, code=_BROKEN,
        ev=_inconclusive(),
        rerun=lambda c: (_ for _ in ()).throw(AssertionError("no rerun")),
        what="verify:h1"))
    assert ev.verdict == "inconclusive" and code == _BROKEN


def test_repaired_code_is_stored_for_replay():
    mem = _memory()
    hyp_id = mem.add_hypothesis("a01", "claim here", {"metric": "precision",
                                                      "op": ">=", "target": 0.9},
                                protocol_code=_BROKEN)
    llm = FakeLLM([f"```python\n{_FIXED}\n```"])

    def rerun(code):
        return Evidence(cmd="x", exit_code=0, verdict="confirmed",
                        method="protocol", output="", meta={})

    asyncio.run(_repair_protocol_rounds(
        _ctx(mem, llm), {}, claim="claim here", criterion=_CRIT,
        code=_BROKEN, ev=_inconclusive(), rerun=rerun,
        what="implement:h1", hyp_id=hyp_id))
    assert mem.get_hypothesis(hyp_id)["protocol_code"] == _FIXED
