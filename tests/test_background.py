"""Tests for --branch: git-branch isolation for unattended runs. Offline —
every git operation runs against a throwaway repo in tmp_path."""

import subprocess

from rich.console import Console

from tythancode import cli
from tythancode.background import (
    commit_leftover_changes,
    is_git_repo,
    slugify,
    start_branch,
)
from tythancode.providers.base import Backend, ToolCall, TurnResult
from tythancode.ui import UI


def make_ui():
    return UI(Console(record=True, force_terminal=False, width=200))


def init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "a.txt").write_text("v1\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def current_branch(path):
    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_slugify_normalizes_free_text():
    assert slugify("Fix Issue #42!") == "fix-issue-42"
    assert slugify("   ") == "task"  # never empty
    assert slugify("a" * 100, max_len=10) == "a" * 10


def test_is_git_repo(tmp_path):
    assert is_git_repo(tmp_path) is False
    init_repo(tmp_path)
    assert is_git_repo(tmp_path) is True


def test_start_branch_not_a_repo_reports_and_returns_none(tmp_path):
    ui = make_ui()
    assert start_branch(tmp_path, None, "do a thing", ui) is None
    assert "needs a git repository" in ui.console.export_text()


def test_start_branch_auto_names_from_prompt(tmp_path):
    init_repo(tmp_path)
    ui = make_ui()
    branch = start_branch(tmp_path, None, "Fix the login bug", ui)
    assert branch is not None
    assert branch.startswith("tythancode/fix-the-login-bug-")
    assert current_branch(tmp_path) == branch


def test_start_branch_uses_explicit_name(tmp_path):
    init_repo(tmp_path)
    ui = make_ui()
    branch = start_branch(tmp_path, "my-feature", "irrelevant", ui)
    assert branch == "my-feature"
    assert current_branch(tmp_path) == "my-feature"


def test_start_branch_keeps_uncommitted_work_in_progress(tmp_path):
    """checkout -b must not discard/stash whatever's already dirty in the
    working tree — the whole point is to isolate an in-progress task."""
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("dirty\n")
    ui = make_ui()
    start_branch(tmp_path, "iso", "task", ui)
    assert (tmp_path / "a.txt").read_text() == "dirty\n"


def test_start_branch_name_collision_reports_and_returns_none(tmp_path):
    init_repo(tmp_path)
    subprocess.run(["git", "branch", "taken"], cwd=tmp_path, check=True)
    ui = make_ui()
    assert start_branch(tmp_path, "taken", "task", ui) is None
    assert "couldn't create branch" in ui.console.export_text()


def test_commit_leftover_changes_noop_when_clean(tmp_path):
    init_repo(tmp_path)
    ui = make_ui()
    commit_leftover_changes(tmp_path, "some-branch", ui)
    assert ui.console.export_text().strip() == ""


def test_commit_leftover_changes_commits_dirty_worktree(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("v2\n")
    (tmp_path / "new.txt").write_text("hi\n")
    ui = make_ui()

    commit_leftover_changes(tmp_path, "my-branch", ui)

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert status.stdout.strip() == ""  # nothing left uncommitted
    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "my-branch" in log.stdout
    assert "committed remaining changes" in ui.console.export_text()


# -- end-to-end through cli.main(): branch created before the run, --------
# -- leftover changes committed after it -----------------------------------


class _OneShotWriteBackend(Backend):
    """Emits a single write_file tool call, then ends the turn."""

    name = "fake"

    def __init__(self):
        super().__init__(model="fake-model")
        self.done = False

    def add_user_message(self, messages, text):
        messages.append({"role": "user", "content": text})

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool_results", "results": results})

    def stream_turn(self, messages, system, tools, ui):
        if not self.done:
            self.done = True
            return TurnResult("tool_use", [ToolCall(
                id="1", name="write_file", input={"path": "new.txt", "content": "hi\n"},
            )])
        return TurnResult("end")


def test_main_with_branch_isolates_run_and_commits_leftovers(tmp_path, monkeypatch):
    init_repo(tmp_path)
    monkeypatch.setattr(cli, "load_provider_configs", lambda: ("fake", {"fake": object()}))
    monkeypatch.setattr(cli, "build_backend", lambda *a, **k: _OneShotWriteBackend())

    rc = cli.main([str(tmp_path), "-p", "add a file", "--branch", "--yolo"])

    assert rc == 0
    assert (tmp_path / "new.txt").read_text() == "hi\n"
    branch = current_branch(tmp_path)
    assert branch.startswith("tythancode/add-a-file-")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert status.stdout.strip() == ""  # the write_file result was committed, nothing left dirty


def test_main_without_branch_flag_does_not_touch_git(tmp_path, monkeypatch):
    init_repo(tmp_path)
    monkeypatch.setattr(cli, "load_provider_configs", lambda: ("fake", {"fake": object()}))
    monkeypatch.setattr(cli, "build_backend", lambda *a, **k: _OneShotWriteBackend())

    rc = cli.main([str(tmp_path), "-p", "add a file", "--yolo"])

    assert rc == 0
    assert current_branch(tmp_path) in ("master", "main")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "new.txt" in status.stdout  # written, but nothing auto-commits it without --branch
