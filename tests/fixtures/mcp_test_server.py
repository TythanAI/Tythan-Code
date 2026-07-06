"""A tiny real MCP server, run as a subprocess by test_mcp_client.py.

Exists so the MCP client is tested against an actual stdio JSON-RPC session
(a real subprocess, real protocol handshake) instead of only mocks — the
thing that's actually hard to get right (task/cancel-scope affinity, the
background event loop, timeouts) can't be verified any other way.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("probe")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def shout(text: str) -> str:
    """Not marked read-only — exercises the default-requires-confirmation path."""
    return text.upper()


@mcp.tool()
def boom() -> str:
    """Always raises, to exercise error propagation back to the caller."""
    raise RuntimeError("boom")


if __name__ == "__main__":
    mcp.run()
