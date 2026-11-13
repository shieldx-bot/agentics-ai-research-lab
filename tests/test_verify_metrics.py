"""P16 — metric adjudication on top of verify_ops.

The criterion is pre-registered OUTSIDE the protocol; the verdict comes from
comparing MEASURED numbers against it. A blocked command, a missing metric
or a non-reproducible protocol can never read as `supported`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.verify_metrics import (
    RESULT_MARKER,
    Criterion,
    adjudicate,
    metrics_agree,
    parse_metrics,
    run_protocol,
)

PY = sys.executable


# ---------------------------------------------------------------- criterion
def test_criterion_rejects_unknown_op():
    with pytest.raises(ValueError):
        Criterion(metric="precision", op="~=", target=0.9)
    with pytest.raises(ValueError):
        Criterion(metric="precision", op=">=", target="not-a-number")


def test_criterion_describe_between():
    c = Criterion(metric="f1", op="between", target=0.8, tolerance=0.95)
    assert c.describe() == "f1 between 0.8 and 0.95"


# ---------------------------------------------------------------- parse
def test_parse_metrics_finds_the_last_marker_line():
    out = (
        "step 1 done\n"
        "noise AGENTICS_RESULT: {\"precision\": 0.5}\n"
        f"final {RESULT_MARKER} {{\"precision\": 0.93, \"n\": 100}}\n"
    )
    assert parse_metrics(out) == {"precision": 0.93, "n": 100}


def test_parse_metrics_returns_empty_when_contract_violated():
    assert parse_metrics("no marker here") == {}
    assert parse_metrics(f"{RESULT_MARKER} not-json") == {}


# ---------------------------------------------------------------- adjudicate
def test_adjudicate_all_basic_ops():
    assert adjudicate({"m": 0.93}, Criterion("m", ">=", 0.9))[0] == "supported"
    assert adjudicate({"m": 0.85}, Criterion("m", ">=", 0.9))[0] == "refuted"
    assert adjudicate({"m": 3}, Criterion("m", "<=", 5))[0] == "supported"
    assert adjudicate({"m": 5}, Criterion("m", "<", 5))[0] == "refuted"
    assert adjudicate({"m": 5}, Criterion("m", ">", 5))[0] == "refuted"
    assert adjudicate({"m": 5}, Criterion("m", "==", 5))[0] == "supported"
    assert adjudicate({"m": 5}, Criterion("m", "!=", 5))[0] == "refuted"
    assert adjudicate({"m": 0.9}, Criterion("m", "between", 0.8, 0.95))[0] == "supported"


def test_adjudicate_missing_or_non_numeric_metric_is_inconclusive():
    verdict, note = adjudicate({"other": 1.0}, Criterion("m", ">=", 0.9))
    assert verdict == "inconclusive" and "not measured" in note
    verdict, note = adjudicate({"m": "high"}, Criterion("m", ">=", 0.9))
    assert verdict == "inconclusive" and "not a number" in note
    assert adjudicate({"m": True}, Criterion("m", ">=", 0.9))[0] == "inconclusive"


def test_adjudicate_coerces_numeric_strings():
    assert adjudicate({"m": "0.93"}, Criterion("m", ">=", 0.9))[0] == "supported"


# ---------------------------------------------------------------- protocol
def _workspace(tmp_path: Path) -> RepoWorkspace:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    ws = RepoWorkspace("a01", settings)
    ws.path.mkdir(parents=True, exist_ok=True)
    return ws


def _script(ws: RepoWorkspace, code: str, name: str = "exp.py") -> str:
    (ws.path / name).write_text(code)
    return name


def test_run_protocol_supported(tmp_path):
    ws = _workspace(tmp_path)
    script = _script(ws, f"print('running...')\nprint({RESULT_MARKER!r} "
                         "+ '{\"precision\": 0.93}')\n")
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.9))
    assert ev.verdict == "confirmed" and ev.ok and ev.method == "protocol"
    assert ev.meta["metrics"] == {"precision": 0.93}
    assert ev.meta["criterion"] == {"metric": "precision", "op": ">=", "target": 0.9}
    assert len(ev.meta["runs"]) == 2                      # rerun like a researcher
    assert "0.93" in ev.output and "measured" in ev.output


def test_run_protocol_refuted(tmp_path):
    ws = _workspace(tmp_path)
    script = _script(ws, f"print({RESULT_MARKER!r} + '{{\"precision\": 0.5}}')\n")
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.9))
    assert ev.verdict == "refuted" and not ev.ok
    assert "measured precision=0.5" in ev.output


def test_run_protocol_non_reproducible_is_inconclusive(tmp_path):
    ws = _workspace(tmp_path)
    code = (
        "import random, json\n"
        "p = round(random.random(), 2)\n"
        "print('AGENTICS_RESULT:', json.dumps({'precision': p}))\n"
    )
    script = _script(ws, code)
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.0))
    assert ev.verdict == "inconclusive" and "reproducible" in ev.output


def test_run_protocol_without_marker_is_inconclusive(tmp_path):
    ws = _workspace(tmp_path)
    script = _script(ws, "print('done', 0.99)\n")
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.9))
    assert ev.verdict == "inconclusive" and RESULT_MARKER in ev.output


def test_run_protocol_timeout_is_inconclusive(tmp_path):
    ws = _workspace(tmp_path)
    script = _script(ws, "import time\ntime.sleep(5)\n")
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.9), timeout=1)
    assert ev.verdict == "inconclusive" and "inconclusive (exit 124)" in ev.output


def test_run_protocol_blocked_path_is_inconclusive(tmp_path):
    ws = _workspace(tmp_path)
    ev = run_protocol(ws, "../outside.py", Criterion("precision", ">=", 0.9))
    assert ev.verdict == "inconclusive" and ev.exit_code == 126


def test_run_protocol_single_run_when_repeat_one(tmp_path):
    ws = _workspace(tmp_path)
    script = _script(ws, f"print({RESULT_MARKER!r} + '{{\"f1\": 1.0}}')\n")
    ev = run_protocol(ws, script, Criterion("f1", "==", 1.0), repeat=1)
    assert ev.verdict == "confirmed" and len(ev.meta["runs"]) == 1


# ------------------------------------------------- reproducibility (noise)
def test_metrics_agree_semantics():
    assert metrics_agree({"p": 0.72}, {"p": 0.72 + 1e-15})      # float noise
    assert metrics_agree({"p": 100}, {"p": 100.0})              # int vs float
    assert not metrics_agree({"p": 0.72}, {"p": 0.69})          # real difference
    assert not metrics_agree({"p": 0.72}, {"q": 0.72})          # different keys
    assert not metrics_agree({"p": 0.72}, {"p": 0.72, "n": 1})  # extra metric
    assert not metrics_agree({"device": "cpu"}, {"device": "gpu"})
    assert metrics_agree({"device": "cpu", "acc": 0.9},
                         {"device": "cpu", "acc": 0.9000000000000001})


def test_run_protocol_float_noise_is_still_reproducible(tmp_path):
    """Same measurement, last-float-bit jitter -> reproducible, not inconclusive.

    json.dumps renders 0.72 and 0.72 + 1e-15 differently, so the old
    bit-identical comparison read a deterministic protocol as
    'not reproducible' — the gate is about the measurement, not the bytes."""
    ws = _workspace(tmp_path)
    code = (
        "import json, pathlib\n"
        "c = pathlib.Path('counter.txt')\n"
        "n = int(c.read_text()) if c.exists() else 0\n"
        "c.write_text(str(n + 1))\n"
        "p = 0.72 if n == 0 else 0.72 + 1e-15\n"
        "print('AGENTICS_RESULT: ' + json.dumps({'precision': p}))\n"
    )
    script = _script(ws, code)
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.7))
    assert ev.verdict == "confirmed" and ev.ok
    assert ev.meta["metrics"]["precision"] == pytest.approx(0.72, rel=1e-12)


def test_run_protocol_real_disagreement_stays_inconclusive(tmp_path):
    ws = _workspace(tmp_path)
    code = (
        "import json, pathlib\n"
        "c = pathlib.Path('counter.txt')\n"
        "n = int(c.read_text()) if c.exists() else 0\n"
        "c.write_text(str(n + 1))\n"
        "p = 0.72 if n == 0 else 0.69\n"
        "print('AGENTICS_RESULT: ' + json.dumps({'precision': p}))\n"
    )
    script = _script(ws, code)
    ev = run_protocol(ws, script, Criterion("precision", ">=", 0.5))
    assert ev.verdict == "inconclusive" and "reproducible" in ev.output


def test_run_protocol_string_metric_flip_is_not_reproducible(tmp_path):
    ws = _workspace(tmp_path)
    code = (
        "import json, pathlib\n"
        "c = pathlib.Path('counter.txt')\n"
        "n = int(c.read_text()) if c.exists() else 0\n"
        "c.write_text(str(n + 1))\n"
        "dev = 'cpu' if n == 0 else 'gpu'\n"
        "print('AGENTICS_RESULT: ' + json.dumps({'acc': 0.9, 'device': dev}))\n"
    )
    script = _script(ws, code)
    ev = run_protocol(ws, script, Criterion("acc", ">=", 0.5))
    assert ev.verdict == "inconclusive" and "reproducible" in ev.output
