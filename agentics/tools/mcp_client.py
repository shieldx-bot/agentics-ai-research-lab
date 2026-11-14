"""MCP bridge (#3 of the advanced-tech roadmap, wave-3): a minimal Model
Context Protocol client for OWNER-APPROVED stdio servers, so the lab can
adopt the ecosystem's servers (arXiv, GitHub, filesystem, ...) without
hand-wrapping each one.

Design (honest add-on, same rules as every module in this stack):

* config-gated — ``tools.mcp_servers`` in configs/lab.yaml (or the
  ``MCP_SERVERS`` env JSON): an empty map is a strict no-op. Roadmap
  boundary: "không tin schema server lạ" — a capability is usable only
  after the OWNER added it to the config;
* stdio JSON-RPC 2.0 only, no SDK dependency: initialize → tools/list →
  tools/call over line-delimited pipes with a selector-based read
  timeout, so a wedged server cannot hang a cycle;
* MCP tools are EXTERNAL capabilities — they land in their own
  ``mcp_tools`` table, NOT ``tool_registry`` (that table means "code
  merged via a PR", and its consumers must keep that meaning);
* the bridge never raises: a failed server journals ``mcp_sync_error``
  and the lab continues exactly as before — ``research/free_tools.py``
  stays the keyless fallback;
* calling a tool is still just DISCOVERY here: any actual execution the
  lab performs keeps passing the ordinary P20 budget + sandbox gates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import selectors
import shlex
import subprocess
import time

log = logging.getLogger("tools.mcp")

_MCP_TIMEOUT_S = 30.0


class McpError(RuntimeError):
    """Any MCP transport/protocol failure (never escapes the bridge)."""


def coerce_command(cmd) -> list[str]:
    """Accept a command string ("python -m foo") or list; raise McpError
    on anything unusable."""
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    elif isinstance(cmd, (list, tuple)):
        parts = [str(c) for c in cmd]
    else:
        raise McpError(f"invalid MCP server command: {cmd!r}")
    if not parts or not parts[0].strip():
        raise McpError("empty MCP server command")
    return parts


class McpStdioClient:
    """One stdio MCP server connection (blocking API; run in a thread)."""

    def __init__(self, command, timeout_s: float = _MCP_TIMEOUT_S) -> None:
        self.command = coerce_command(command)
        self.timeout_s = float(timeout_s)
        self._proc: subprocess.Popen | None = None
        self._next_id = 0

    def __enter__(self) -> "McpStdioClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ transport
    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _readline(self, remaining: float) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            ready = sel.select(timeout=max(0.0, remaining))
            if not ready:
                raise McpError(
                    f"timeout after {self.timeout_s:.0f}s waiting for the "
                    "MCP response")
            line = self._proc.stdout.readline()
        finally:
            sel.close()
        if not line.strip():
            raise McpError("MCP server closed stdout or sent an empty line")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise McpError(f"non-JSON MCP line: {exc}") from exc
        if not isinstance(payload, dict):
            raise McpError("non-object MCP payload")
        return payload

    def _rpc(self, method: str, params: dict | None = None,
             notify: bool = False):
        if self._proc is None or self._proc.stdin is None:
            raise McpError("MCP client not started")
        self._next_id += 1
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self._next_id
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise McpError(f"cannot write to the MCP server: {exc}") from exc
        if notify:
            return None
        deadline = time.monotonic() + self.timeout_s
        while True:                       # skip interleaved notifications
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(f"timeout waiting for the {method} response")
            payload = self._readline(remaining)
            if payload.get("id") == self._next_id:
                if "error" in payload:
                    raise McpError(f"{method}: {payload['error']}")
                return payload.get("result")

    # ------------------------------------------------------------- protocol
    def initialize(self) -> dict:
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agentics-lab", "version": "1.0"},
        })
        try:
            self._rpc("notifications/initialized", notify=True)
        except Exception:  # noqa: BLE001 — the notification never fails a handshake
            pass
        return result if isinstance(result, dict) else {}

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list", {}) or {}
        tools = result.get("tools") if isinstance(result, dict) else None
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._rpc("tools/call",
                           {"name": name, "arguments": arguments or {}})
        if not isinstance(result, dict):
            return {"ok": False, "text": "", "raw": result}
        content = result.get("content") or []
        text = "\n".join(str(c.get("text", ""))
                         for c in content if isinstance(c, dict))
        return {"ok": not bool(result.get("isError")),
                "text": text, "raw": result}


# ------------------------------------------------------------------ bridge
def _sync_one(server: str, command, memory, timeout_s: float) -> int:
    """One server: discover tools and record them. Returns tool count."""
    with McpStdioClient(command, timeout_s=timeout_s) as client:
        client.initialize()
        tools = client.list_tools()
    cleaned = [
        {"name": str(t["name"]),
         "description": str(t.get("description") or "")[:500]}
        for t in tools
        if isinstance(t, dict) and str(t.get("name") or "").strip()
    ]
    return memory.upsert_mcp_tools(server, cleaned)


async def sync_bridge(settings, memory) -> dict:
    """Discover + record capabilities for every configured server.

    Returns {server: tool_count} (negative count = failed). Never raises;
    runs each blocking client in a thread so the event loop never stalls.
    """
    servers = getattr(settings, "mcp_servers", None)
    if not isinstance(servers, dict) or not servers:
        return {}
    timeout_s = float(getattr(settings, "mcp_timeout_s", _MCP_TIMEOUT_S)
                      or _MCP_TIMEOUT_S)
    stats: dict[str, int] = {}
    for server, command in servers.items():
        name = str(server)
        try:
            n = await asyncio.to_thread(_sync_one, name, command, memory,
                                        timeout_s)
            stats[name] = n
            log.info("MCP server %s: %d tool(s) registered", name, n)
        except Exception as exc:  # noqa: BLE001 — one bad server ≠ broken lab
            stats[name] = -1
            log.warning("MCP server %s sync failed: %s: %s",
                        name, type(exc).__name__, exc)
            try:
                memory.journal("system", "mcp_sync_error",
                               {"server": name,
                                "why": f"{type(exc).__name__}: {exc}"[:200]})
            except Exception:  # noqa: BLE001
                pass
    try:
        memory.journal("system", "mcp_sync", dict(stats))
    except Exception:  # noqa: BLE001
        pass
    return stats
