"""MCP client tests against a real (tiny) server subprocess — see
tests/fixtures/mcp_test_server.py. Deliberately not mocked: the things that
are actually easy to get wrong here (the background event loop, task/cancel
-scope affinity, timeouts) can't be verified by mocking away the SDK.
"""

import sys
from pathlib import Path

import pytest

from tythancode.mcp_client import MCPManager, MCPServerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


@pytest.fixture(scope="module")
def manager():
    cfg = MCPServerConfig(name="probe", command=sys.executable, args=[str(FIXTURE)])
    m = MCPManager([cfg])
    m.connect_all()
    yield m
    m.close()


def test_connects_and_lists_tools(manager):
    assert manager.startup_errors == {}
    assert manager.connected_servers == ["probe"]
    names = {t.qualified_name for t in manager.tools}
    assert names == {"mcp__probe__add", "mcp__probe__shout", "mcp__probe__boom"}


def test_read_only_hint_is_picked_up(manager):
    add_tool = manager.find_tool("mcp__probe__add")
    shout_tool = manager.find_tool("mcp__probe__shout")
    assert add_tool.read_only is True
    assert shout_tool.read_only is False


def test_tool_definition_shape(manager):
    add_tool = manager.find_tool("mcp__probe__add")
    definition = add_tool.to_definition()
    assert definition["name"] == "mcp__probe__add"
    assert "probe" in definition["description"]
    assert definition["input_schema"]["required"] == ["a", "b"]


def test_call_tool_success(manager):
    output, is_error = manager.call_tool("mcp__probe__add", {"a": 2, "b": 3})
    assert is_error is False
    assert "5" in output


def test_call_tool_that_raises_is_reported_as_error(manager):
    output, is_error = manager.call_tool("mcp__probe__boom", {})
    assert is_error is True
    assert "boom" in output.lower()


def test_unknown_tool(manager):
    output, is_error = manager.call_tool("mcp__probe__nope", {})
    assert is_error is True
    assert "Unknown tool" in output


def test_empty_manager_is_a_safe_no_op():
    m = MCPManager([])
    m.connect_all()  # must not spin up a thread or block
    assert m.tools == []
    assert m.find_tool("mcp__anything__x") is None
    m.close()  # must not raise on a manager that never connected


def test_server_that_fails_to_start_is_isolated():
    bad = MCPServerConfig(name="bad", command="this-command-does-not-exist-xyz")
    m = MCPManager([bad])
    m.connect_all()
    assert m.tools == []
    assert "bad" in m.startup_errors
    m.close()  # must not hang waiting to shut down a session that never started


def test_close_stops_the_background_thread(manager_to_close=None):
    cfg = MCPServerConfig(name="probe2", command=sys.executable, args=[str(FIXTURE)])
    m = MCPManager([cfg])
    m.connect_all()
    thread = m._thread
    assert thread.is_alive()
    m.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
