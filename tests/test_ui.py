"""UI rendering/interaction tests that don't fit test_cli.py's slash-command
focus — currently just the per-hunk review prompt's y/n/a/d parsing."""

from rich.console import Console

from tythancode.hunks import split_hunks
from tythancode.ui import UI


def make_ui():
    return UI(Console(force_terminal=False))


def scripted_input(answers):
    it = iter(answers)

    def fake(prompt=""):
        return next(it)

    return fake


def test_review_hunks_yes_and_no(monkeypatch):
    old = "a\nb\nc\nd\ne\n"
    new = "A\nb\nc\nD\ne\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 2

    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["n", "y"]))
    accepted = ui.review_hunks("a.txt", hunks)
    assert accepted == [False, True]


def test_review_hunks_accept_all_remaining(monkeypatch):
    old = "a\nb\nc\nd\ne\nf\ng\n"
    new = "A\nb\nC\nd\nE\nf\nG\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 4  # a, c, e, g each change in isolation

    ui = make_ui()
    # "a" on the first hunk should apply it and every remaining one too,
    # without prompting again.
    monkeypatch.setattr(ui.console, "input", scripted_input(["a"]))
    accepted = ui.review_hunks("a.txt", hunks)
    assert accepted == [True, True, True, True]


def test_review_hunks_decline_all_remaining(monkeypatch):
    old = "a\nb\nc\nd\ne\nf\ng\n"
    new = "A\nb\nC\nd\nE\nf\nG\n"
    hunks = split_hunks(old, new)
    assert len(hunks) == 4

    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["d"]))
    accepted = ui.review_hunks("a.txt", hunks)
    assert accepted == [False, False, False, False]


def test_review_hunks_empty_answer_defaults_to_no(monkeypatch):
    old, new = "a\nb\n", "A\nb\n"
    hunks = split_hunks(old, new)
    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input([""]))
    assert ui.review_hunks("a.txt", hunks) == [False]


def test_review_hunks_reprompts_on_garbage_input(monkeypatch):
    old, new = "a\nb\n", "A\nb\n"
    hunks = split_hunks(old, new)
    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["blah", "???", "y"]))
    assert ui.review_hunks("a.txt", hunks) == [True]
