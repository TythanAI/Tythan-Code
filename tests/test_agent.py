"""Agent-loop tests with a stubbed backend (no network)."""

import sys
from pathlib import Path

import pytest

from tythancode.agent import Agent
from tythancode.config import Config
from tythancode.mcp_client import MCPManager, MCPServerConfig
from tythancode.providers.base import Backend, ToolCall, TurnResult
from tythancode.ui import UI

from rich.console import Console

MCP_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


class FakeBackend(Backend):
    """Scripted backend: returns pre-baked TurnResults, records history calls."""

    name = "fake"

    def __init__(self, turns: list[TurnResult]):
        super().__init__(model="fake-model")
        self.turns = list(turns)
        self.calls = 0

    def add_user_message(self, messages, text):
        messages.append({"role": "user", "content": text})

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool_results", "results": results})

    def stream_turn(self, messages, system, tools, ui):
        self.calls += 1
        assert "Tythan Code" in system  # system prompt is passed through
        assert any(t["name"] == "read_file" for t in tools)
        return self.turns.pop(0)


def make_agent(tmp_path, turns, yolo=True, mcp_manager=None):
    config = Config(workspace=tmp_path, yolo=yolo)
    ui = UI(Console(file=open("/dev/null", "w"), force_terminal=False))
    return Agent(config, ui, FakeBackend(turns), mcp_manager=mcp_manager,
                 code_index_cache_dir=tmp_path / ".index-cache")


@pytest.fixture(scope="module")
def connected_mcp():
    cfg = MCPServerConfig(name="probe", command=sys.executable, args=[str(MCP_FIXTURE)])
    manager = MCPManager([cfg])
    manager.connect_all()
    yield manager
    manager.close()


def test_tool_round_trip(tmp_path):
    (tmp_path / "hello.txt").write_text("hi there\n")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="read_file", input={"path": "hello.txt"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("what's in hello.txt?")

    assert agent.backend.calls == 2
    # history: user, tool_results (assistant msgs are appended by real backends)
    tool_msg = agent.messages[1]
    assert tool_msg["role"] == "tool_results"
    result = tool_msg["results"][0]
    assert result.call_id == "tu_1"
    assert "hi there" in result.output
    assert result.is_error is False


def test_tool_error_reported(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="read_file", input={"path": "missing.txt"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("read missing.txt")

    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "not found" in result.output


def test_refusal_ends_turn(tmp_path):
    agent = make_agent(tmp_path, [TurnResult("refusal")])
    agent.run_turn("hello")
    assert agent.backend.calls == 1
    assert agent.messages[0]["role"] == "user"
    assert len(agent.messages) == 1


def test_write_declined_when_not_confirmed(tmp_path, monkeypatch):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                         input={"path": "a.txt", "content": "data"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.config.yolo = False
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: False)
    monkeypatch.setattr(agent.ui, "show_diff", lambda *a, **k: None)

    agent.run_turn("create a.txt")

    assert not (tmp_path / "a.txt").exists()
    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "declined" in result.output


def test_system_prompt_includes_project_rules(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Always run `make test` before finishing.\n", encoding="utf-8")
    agent = make_agent(tmp_path, [])
    prompt = agent.system_prompt()
    assert "AGENTS.md" in prompt
    assert "Always run `make test` before finishing." in prompt


def test_system_prompt_unchanged_without_rules_file(tmp_path):
    agent = make_agent(tmp_path, [])
    assert agent.project_rules is None
    assert "Project-specific instructions" not in agent.system_prompt()


def test_todo_write_updates_agent_state(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="todo_write", input={
            "todos": [
                {"content": "read the file", "status": "completed"},
                {"content": "fix the bug", "status": "in_progress"},
            ]
        })]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("fix the bug")

    assert agent.todos == [
        {"content": "read the file", "status": "completed"},
        {"content": "fix the bug", "status": "in_progress"},
    ]
    result = agent.messages[1]["results"][0]
    assert result.is_error is False
    assert "2 item(s)" in result.output


def test_todo_write_invalid_input_reports_tool_error(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="todo_write", input={
            "todos": [{"content": "x", "status": "not-a-status"}]
        })]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("do something")

    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "invalid status" in result.output
    assert agent.todos == []  # rejected input never replaces existing state


def test_reset_clears_todos(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="todo_write", input={
            "todos": [{"content": "x", "status": "pending"}]
        })]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("plan it")
    assert agent.todos

    agent.reset()
    assert agent.todos == []


def test_fetch_url_requires_confirmation(tmp_path, monkeypatch):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="fetch_url", input={"url": "https://example.com"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns, yolo=False)
    asked = []
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: (asked.append(prompt), False)[1])

    agent.run_turn("check the docs")

    assert asked
    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "declined" in result.output


def test_fetch_url_success_once_confirmed(tmp_path, monkeypatch):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="fetch_url", input={"url": "https://example.com"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns, yolo=False)
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: True)
    monkeypatch.setattr("tythancode.webfetch.fetch_url", lambda url: "page content")

    agent.run_turn("check the docs")

    result = agent.messages[1]["results"][0]
    assert result.is_error is False
    assert result.output == "page content"


def test_fetch_url_skips_confirmation_in_yolo_mode(tmp_path, monkeypatch):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="fetch_url", input={"url": "https://example.com"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns, yolo=True)
    monkeypatch.setattr(agent.ui, "confirm", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("yolo mode should not ask for confirmation")))
    monkeypatch.setattr("tythancode.webfetch.fetch_url", lambda url: "page content")

    agent.run_turn("check the docs")

    result = agent.messages[1]["results"][0]
    assert result.is_error is False


def test_fetch_url_error_reported_as_tool_error(tmp_path, monkeypatch):
    from tythancode.webfetch import FetchError

    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="fetch_url", input={"url": "http://localhost/"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns, yolo=True)

    def raise_fetch_error(url):
        raise FetchError("Refusing to fetch a private/internal/loopback address")

    monkeypatch.setattr("tythancode.webfetch.fetch_url", raise_fetch_error)

    agent.run_turn("check localhost")

    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "private/internal/loopback" in result.output


def test_code_search_tool_returns_relevant_file(tmp_path):
    (tmp_path / "net.py").write_text("def retry_with_backoff(fn):\n    pass\n", encoding="utf-8")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="code_search", input={"query": "retry backoff"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("where is retry logic?")

    result = agent.messages[1]["results"][0]
    assert result.is_error is False
    assert "net.py" in result.output


def test_code_search_index_is_cached_across_calls(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="code_search", input={"query": "foo"})]),
        TurnResult("tool_use", [ToolCall(id="tu_2", name="code_search", input={"query": "foo"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, turns)
    agent.run_turn("find foo")
    index_after_first = agent._code_index
    assert index_after_first is not None
    assert agent._get_code_index() is index_after_first


def test_write_file_invalidates_code_index(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    agent = make_agent(tmp_path, [])
    agent._get_code_index()
    assert agent._code_index is not None

    output, is_error = agent._execute_tool("write_file", {"path": "b.py", "content": "def bar(): pass\n"})
    assert is_error is False
    assert agent._code_index is None


def test_reindex_cache_survives_a_new_agent_instance(tmp_path, monkeypatch):
    """The whole point of a persistent (disk) cache, as opposed to the
    in-memory one Agent already had: a brand-new Agent — standing in for a
    fresh Tythan Code process after a restart — reuses it too."""
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    agent1 = make_agent(tmp_path, [])
    index1 = agent1.reindex()
    assert any(c.path == "a.py" for c in index1.chunks)

    import tythancode.codeindex as codeindex
    calls = []
    original = codeindex._chunk_file

    def spy(rel_path, text):
        calls.append(rel_path)
        return original(rel_path, text)

    monkeypatch.setattr(codeindex, "_chunk_file", spy)

    agent2 = make_agent(tmp_path, [])
    index2 = agent2.reindex()
    assert calls == []  # reused agent1's on-disk cache instead of re-tokenizing
    assert [c.text for c in index2.chunks] == [c.text for c in index1.chunks]


def test_tool_definitions_include_connected_mcp_tools(tmp_path, connected_mcp):
    agent = make_agent(tmp_path, [], mcp_manager=connected_mcp)
    names = {d["name"] for d in agent.tool_definitions()}
    assert "read_file" in names  # built-ins still present
    assert "mcp__probe__add" in names
    assert "mcp__probe__shout" in names


def test_mcp_read_only_tool_runs_without_confirmation(tmp_path, connected_mcp, monkeypatch):
    agent = make_agent(tmp_path, [], yolo=False, mcp_manager=connected_mcp)
    monkeypatch.setattr(agent.ui, "confirm", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not ask to confirm a read-only MCP tool")))
    output, is_error = agent._execute_tool("mcp__probe__add", {"a": 2, "b": 3})
    assert is_error is False
    assert "5" in output


def test_mcp_non_read_only_tool_requires_confirmation(tmp_path, connected_mcp, monkeypatch):
    agent = make_agent(tmp_path, [], yolo=False, mcp_manager=connected_mcp)
    asked = []
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: (asked.append(prompt), False)[1])

    output, is_error = agent._execute_tool("mcp__probe__shout", {"text": "hi"})

    assert asked  # confirmation was actually requested
    assert is_error is True
    assert "declined" in output


def test_mcp_non_read_only_tool_runs_once_confirmed(tmp_path, connected_mcp, monkeypatch):
    agent = make_agent(tmp_path, [], yolo=False, mcp_manager=connected_mcp)
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: True)

    output, is_error = agent._execute_tool("mcp__probe__shout", {"text": "hi"})

    assert is_error is False
    assert "HI" in output


def test_mcp_tool_error_is_reported_as_tool_error(tmp_path, connected_mcp):
    agent = make_agent(tmp_path, [], mcp_manager=connected_mcp)
    output, is_error = agent._execute_tool("mcp__probe__boom", {})
    assert is_error is True
    assert "boom" in output.lower()


def test_unknown_mcp_tool_reported_cleanly(tmp_path):
    agent = make_agent(tmp_path, [])  # no MCP manager configured at all
    output, is_error = agent._execute_tool("mcp__nope__nope", {})
    assert is_error is True
    assert "Unknown tool" in output


def test_set_backend_resets_history(tmp_path):
    agent = make_agent(tmp_path, [TurnResult("end")])
    agent.run_turn("hi")
    assert agent.messages
    agent.set_backend(FakeBackend([TurnResult("end")]))
    assert agent.messages == []
