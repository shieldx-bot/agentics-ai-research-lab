"""Owner visibility: the `agentics panel` terminal reader (P19).

The dashboard already shows the panel column for open MRs; this CLI command
is the owner's direct line from a terminal — live panels, one MR's full vote
history, and the mechanism's own performance numbers.
"""

from typer.testing import CliRunner

from agentics import cli
from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory
from agentics.peer_review import PeerPanel


def _setup(tmp_path) -> tuple[Settings, SharedMemory]:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    mem = SharedMemory(s.db_path)
    mem.heartbeat("coder-1", "coder", 1)
    mem.heartbeat("rev-a", "coder", 1)
    mem.heartbeat("rev-b", "writer", 1)
    return s, mem


def _approved_after_defense(mem) -> int:
    mr = mem.open_review(task_id=None, title="good MR", body="b", author="coder-1",
                         branch="agent/coder-1/g", files=["a.py"])
    p = PeerPanel(mem, panel_size=2)
    p.cast_vote(mr, "rev-a", "reject", "metric not measured")
    p.cast_vote(mr, "rev-b", "accept", "fine")
    p.submit_defense(mr, "coder-1", {"rev-a": "measured again, passes"}, note="ok")
    p.cast_vote(mr, "rev-a", "accept", "fair")
    return mr


def _rejected_on_deadline(mem) -> int:
    mr = mem.open_review(task_id=None, title="late MR", body="b", author="rev-b",
                         branch="agent/rev-b/l", files=["b.py"])
    p = PeerPanel(mem, panel_size=2)
    p.cast_vote(mr, "rev-a", "reject", "no evidence")
    p.concede(mr, "rev-b", "the objection is right")
    return mr


def test_panel_cli_lists_panels(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    _approved_after_defense(mem)
    _rejected_on_deadline(mem)
    # a live panel still voting — its TITLE is what the table shows
    live = mem.open_review(task_id=None, title="open MR", body="b", author="coder-1",
                           branch="agent/coder-1/o", files=["c.py"])
    mem.ensure_panel(live)
    mem.record_panel_vote(live, 1, "rev-a", "accept", "early accept")
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel"])
    assert result.exit_code == 0, result.output
    # closed panels show their verdict line, the live one shows its title
    for needle in ("open MR", "voting", "approved", "rejected", "coder-1",
                   "unanimous", "conceded"):
        assert needle in result.output
    mem.close()


def test_panel_cli_state_filter_and_empty(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel", "--state", "voting"])
    assert result.exit_code == 0, result.output
    assert "chưa có panel nào" in result.output
    mem.close()


def test_panel_cli_mr_detail_shows_votes_objections_defense(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    mr = _approved_after_defense(mem)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel", "--mr", str(mr)])
    assert result.exit_code == 0, result.output
    # round 1 holds ONLY the decisive reject (the defense opened immediately,
    # so rev-b's late round-1 vote is dropped — the engine's one-reject rule)
    for needle in ("good MR", "vòng 1", "vòng 2", "reject", "accept", "rev-a",
                   "metric not measured", "measured again", "approved"):
        assert needle in result.output
    mem.close()


def test_panel_cli_mr_without_panel(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel", "--mr", "999"])
    assert result.exit_code == 0, result.output
    assert "chưa có panel" in result.output
    mem.close()


def test_panel_cli_stats_reports_mechanism_performance(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    _approved_after_defense(mem)          # approved AFTER defense -> saved
    _rejected_on_deadline(mem)            # conceded -> final reject
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel", "--stats"])
    assert result.exit_code == 0, result.output
    for needle in ("approved: 1", "rejected: 1", "tỉ lệ duyệt", "50%",
                   "bảo vệ cứu được", "conceded: 1", "rev-a"):
        assert needle in result.output
    mem.close()


def test_panel_cli_stats_with_no_panels(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["panel", "--stats"])
    assert result.exit_code == 0, result.output
    assert "chưa có panel nào" in result.output
    mem.close()
