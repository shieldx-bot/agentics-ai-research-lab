"""Wave-3 #3 — MCP bridge (tools/mcp_client.py).

Contract under test:
* command coercion (str / list / garbage);
* a real stdio round-trip against a stub MCP server (initialize →
  tools/list → tools/call) with the selector-based timeout;
* the bridge records discovered capabilities into `mcp_tools`, drops
  tools the server stopped exposing, and journals the sync;
* an empty config is a strict no-op and a broken server never raises.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.tools import mcp_client as mc  # noqa: E402

_STUB = r'''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue                      # notification: no reply
    method = msg.get("method", "")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {},
                  "serverInfo": {"name": "stub", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "echo", "description": "Echoes text back"},
            {"name": "noop", "description": ""}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text",
                               "text": "echo:" + str(msg["params"].get("arguments"))}],
                  "isError": False}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                          "error": {"code": -32601, "message": "unimplemented"}}),
              flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}),
          flush=True)
'''


def _stub_script(tmp_path) -> Path:
    p = tmp_path / "mcp_stub_server.py"
    p.write_text(_STUB, encoding="utf-8")
    return p


# ------------------------------------------------------------- primitives
def test_coerce_command():
    assert mc.coerce_command("python -m foo") == ["python", "-m", "foo"]
    assert mc.coerce_command(["a", 1]) == ["a", "1"]
    try:
        mc.coerce_command("   ")
        assert False, "empty command must raise"
    except mc.McpError:
        pass
    try:
        mc.coerce_command(42)
        assert False, "garbage command must raise"
    except mc.McpError:
        pass


def test_stdio_round_trip(tmp_path):
    stub = _stub_script(tmp_path)
    with mc.McpStdioClient([sys.executable, str(stub)], timeout_s=20) as c:
        info = c.initialize()
        assert info.get("serverInfo", {}).get("name") == "stub"
        tools = c.list_tools()
        names = {t["name"] for t in tools}
        assert {"echo", "noop"} <= names
        out = c.call_tool("echo", {"x": 1})
        assert out["ok"] is True
        assert "echo:" in out["text"]


# ------------------------------------------------------------------ bridge
def _mem(tmp_path):
    return SharedMemory(tmp_path / "mcp.db")


def test_sync_bridge_registers_and_drops(tmp_path):
    stub = _stub_script(tmp_path)
    m = _mem(tmp_path)
    settings = SimpleNamespace(
        mcp_servers={"stub": [sys.executable, str(stub)]},
        mcp_timeout_s=20.0)
    stats = asyncio.run(mc.sync_bridge(settings, m))
    assert stats == {"stub": 2}
    rows = m.list_mcp_tools("stub")
    assert {r["name"] for r in rows} == {"echo", "noop"}
    # second sync with one tool gone → the listing is the truth
    short = tmp_path / "mcp_stub_server_short.py"
    short.write_text(_STUB.replace('{"name": "noop", "description": ""}', ""),
                     encoding="utf-8")
    settings2 = SimpleNamespace(mcp_servers={"stub": [sys.executable, str(short)]},
                                mcp_timeout_s=20.0)
    stats2 = asyncio.run(mc.sync_bridge(settings2, m))
    assert stats2 == {"stub": 1}
    assert {r["name"] for r in m.list_mcp_tools("stub")} == {"echo"}
    events = m._conn.execute(
        "SELECT detail FROM journal WHERE event='mcp_sync'").fetchall()
    assert events, "the sync must be journalled"


def test_sync_bridge_empty_config_noop(tmp_path):
    m = _mem(tmp_path)
    assert asyncio.run(mc.sync_bridge(SimpleNamespace(mcp_servers={}), m)) == {}
    assert asyncio.run(mc.sync_bridge(SimpleNamespace(mcp_servers=None), m)) == {}
    assert m.list_mcp_tools() == []


def test_sync_bridge_broken_server_never_raises(tmp_path):
    m = _mem(tmp_path)
    settings = SimpleNamespace(
        mcp_servers={"broken": [sys.executable, "-c", "import sys; sys.exit(3)"]},
        mcp_timeout_s=10.0)
    stats = asyncio.run(mc.sync_bridge(settings, m))
    assert stats == {"broken": -1}
    errs = m._conn.execute(
        "SELECT detail FROM journal WHERE event='mcp_sync_error'").fetchall()
    assert errs
