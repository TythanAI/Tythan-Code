"""MCP (Model Context Protocol) client: connect to external tool servers
configured in `~/.tythancode/config.json` and expose their tools to the
agent loop alongside the built-in ones.

Tools are namespaced as `mcp__<server>__<tool>` so multiple servers can
never collide on a tool name, and both the model and the user can see which
server a call goes to.

The agent loop, tools and UI are synchronous throughout; the `mcp` SDK is
asyncio-based, and each server connection is a long-lived resource (a stdio
subprocess + a JSON-RPC session) that must survive across many tool calls —
a plain `asyncio.run(...)` per call won't work, since it would tear the
subprocess down and reconnect on every single call. More fundamentally,
`stdio_client`/`ClientSession` are built on anyio cancel scopes, which must
be entered and exited by the exact same asyncio Task; a context manager
opened in one `asyncio.run()` cannot be closed from another. So each
server's whole session — connect, serve calls, disconnect — runs inside one
long-lived coroutine on one background-thread event loop that lives for the
process's whole lifetime. The synchronous side talks to it by dropping
requests on that coroutine's inbox queue and blocking on the resulting
`asyncio.Future` via `run_coroutine_threadsafe(...).result()`.

A server that fails to start (bad command, times out, crashes on
`initialize`) is recorded in `startup_errors` and simply contributes no
tools — it never prevents the other configured servers, or the rest of
Tythan Code, from working.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field

CONNECT_TIMEOUT = 20.0
CALL_TIMEOUT = 120.0


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class MCPTool:
    server: str
    name: str  # bare tool name, as the server itself sees it
    qualified_name: str  # "mcp__<server>__<name>", what the model sees
    description: str
    input_schema: dict
    read_only: bool  # True only if the server explicitly said so (readOnlyHint)

    def to_definition(self) -> dict:
        return {
            "name": self.qualified_name,
            "description": f"[MCP: {self.server}] {self.description}",
            "input_schema": self.input_schema,
        }


class _ServerSession:
    """Owns one server's connection. `run()` must be scheduled once onto the
    background loop and left alone until `shutdown()` — see module
    docstring for why it can't be split across separate loop calls."""

    def __init__(self, cfg: MCPServerConfig):
        self.cfg = cfg
        self.tools: list[MCPTool] = []
        self.ready = asyncio.Event()
        self.error: str | None = None
        self._inbox: "asyncio.Queue[tuple[str, dict, asyncio.Future] | None]" = asyncio.Queue()

    async def run(self) -> None:
        # Imported lazily so merely importing this module doesn't pull in
        # the mcp package's own import-time cost for a session that never
        # configures any MCP servers.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        try:
            params = StdioServerParameters(command=self.cfg.command, args=self.cfg.args, env=self.cfg.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=CONNECT_TIMEOUT)
                    self.tools = [
                        MCPTool(
                            server=self.cfg.name,
                            name=t.name,
                            qualified_name=f"mcp__{self.cfg.name}__{t.name}",
                            description=t.description or "",
                            input_schema=t.inputSchema or {"type": "object", "properties": {}},
                            read_only=bool(getattr(t.annotations, "readOnlyHint", False)) if t.annotations else False,
                        )
                        for t in listed.tools
                    ]
                    self.ready.set()
                    while True:
                        item = await self._inbox.get()
                        if item is None:
                            break
                        tool_name, arguments, future = item
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, arguments), timeout=CALL_TIMEOUT
                            )
                            if not future.cancelled():
                                future.set_result(result)
                        except Exception as exc:
                            if not future.cancelled():
                                future.set_exception(exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ready.set()  # unblock a waiter even when startup itself failed

    async def call(self, tool_name: str, arguments: dict) -> object:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._inbox.put((tool_name, arguments, future))
        return await future

    async def shutdown(self) -> None:
        await self._inbox.put(None)


class MCPManager:
    """Owns the background event-loop thread and every configured server's
    session. Safe and cheap to construct with an empty list — `connect_all`
    then does nothing and every lookup finds no tools, so Tythan Code
    behaves exactly as it did before MCP support existed when no servers
    are configured."""

    def __init__(self, configs: list[MCPServerConfig]):
        self.configs = configs
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, _ServerSession] = {}
        self.startup_errors: dict[str, str] = {}

    @property
    def tools(self) -> list[MCPTool]:
        return [t for s in self._sessions.values() for t in s.tools]

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, s in self._sessions.items() if s.error is None]

    def find_tool(self, qualified_name: str) -> MCPTool | None:
        for tool in self.tools:
            if tool.qualified_name == qualified_name:
                return tool
        return None

    def connect_all(self) -> None:
        """Start every configured server and block until each has either
        connected or failed. Call once, before the first turn."""
        if not self.configs:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        for cfg in self.configs:
            session = _ServerSession(cfg)
            self._sessions[cfg.name] = session
            asyncio.run_coroutine_threadsafe(session.run(), self._loop)

        for name, session in self._sessions.items():
            try:
                self._run(session.ready.wait(), timeout=CONNECT_TIMEOUT + 5)
            except concurrent.futures.TimeoutError:
                self.startup_errors[name] = "timed out waiting for the server to start"
                continue
            if session.error:
                self.startup_errors[name] = session.error

    def call_tool(self, qualified_name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (output_text, is_error) — the same shape every other tool
        executor in agent.py returns."""
        tool = self.find_tool(qualified_name)
        if tool is None:
            return f"Unknown tool: {qualified_name}", True
        session = self._sessions[tool.server]
        try:
            result = self._run(session.call(tool.name, arguments), timeout=CALL_TIMEOUT + 5)
        except concurrent.futures.TimeoutError:
            return f"MCP call to {qualified_name} timed out after {CALL_TIMEOUT:.0f}s", True
        except Exception as exc:
            return f"MCP call to {qualified_name} failed: {type(exc).__name__}: {exc}", True
        return _format_result(result)

    def close(self) -> None:
        if self._loop is None:
            return
        for session in self._sessions.values():
            if session.error is None:  # a session that never connected has no task to stop
                try:
                    self._run(session.shutdown(), timeout=5)
                except Exception:
                    pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


def _format_result(result) -> tuple[str, bool]:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{type(block).__name__} content omitted]")
    text = "\n".join(parts) if parts else "(no content)"
    return text, bool(getattr(result, "isError", False))
