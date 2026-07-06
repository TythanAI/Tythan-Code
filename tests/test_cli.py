"""Tests for cli.py's slash-command handling. Offline — reuses the
ScriptedBackend/make_agent fixtures from test_agent_context.py."""

from rich.console import Console

from tythancode.cli import handle_slash
from tythancode.ui import UI

from test_agent_context import ScriptedBackend, make_agent


def test_model_command_notes_context_window_is_unchanged(tmp_path, capsys):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    handle_slash("/model some-other-model", agent, agent.ui, {})

    assert agent.backend.model == "some-other-model"
    out = console.export_text()
    assert "context window stays" in out
    assert "/context" in out


def test_checkpoints_command_respects_limit_argument(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    for i in range(5):
        agent.checkpoints.begin_turn(f"edit {i}")
        target = tmp_path / "a.txt"
        target.write_text(f"v{i}")
        agent.checkpoints.record_before(target)
        target.write_text(f"v{i + 1}")
        agent.checkpoints.commit_turn()

    handle_slash("/checkpoints 2", agent, agent.ui, {})
    out = console.export_text()
    assert "showing 2 of 5" in out


def test_checkpoints_command_ignores_invalid_limit_argument(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    agent.checkpoints.begin_turn("edit")
    target = tmp_path / "a.txt"
    target.write_text("v1")
    agent.checkpoints.record_before(target)
    target.write_text("v2")
    agent.checkpoints.commit_turn()

    # Non-numeric argument must fall back to the default rather than crash.
    handle_slash("/checkpoints not-a-number", agent, agent.ui, {})
    out = console.export_text()
    assert "edit" in out


def test_undo_command_reports_skipped_binary_files(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    agent.checkpoints.begin_turn("touch files")
    changed = tmp_path / "a.txt"
    changed.write_text("v1")
    agent.checkpoints.record_before(changed)
    changed.write_text("v2")
    binary = tmp_path / "legacy.txt"
    binary.write_bytes(b"caf\xe9")
    agent.checkpoints.record_before(binary)
    agent.checkpoints.commit_turn()

    handle_slash("/undo", agent, agent.ui, {})
    out = console.export_text()
    assert "reverted 1 file(s)" in out
    assert "weren't covered" in out


def test_todos_command_shows_current_plan(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)
    agent.todos = [{"content": "ship it", "status": "in_progress"}]

    handle_slash("/todos", agent, agent.ui, {})
    out = console.export_text()
    assert "ship it" in out
    assert "0/1 done" in out


def test_todos_command_handles_empty_plan(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    handle_slash("/todos", agent, agent.ui, {})
    assert "todo list cleared" in console.export_text()


def test_reindex_command_reports_counts(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    handle_slash("/reindex", agent, agent.ui, {})
    out = console.export_text()
    assert "indexed 1 file(s)" in out
    assert "chunk(s)" in out


def test_undo_command_reports_nothing_when_store_is_empty(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    console = Console(record=True)
    agent.ui = UI(console)

    handle_slash("/undo", agent, agent.ui, {})
    assert "nothing to undo" in console.export_text()
