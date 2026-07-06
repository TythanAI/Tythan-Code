"""UI rendering/interaction tests that don't fit test_cli.py's slash-command
focus — the per-hunk review prompt's y/n/a/d parsing, and the multi-file
batch review that walks every file's hunks as one continuous stream."""

from rich.console import Console

from tythancode.hunks import split_hunks
from tythancode.ui import UI, FileReview


def make_ui():
    return UI(Console(force_terminal=False))


def scripted_input(answers):
    it = iter(answers)

    def fake(prompt=""):
        return next(it)

    return fake


def make_review(call_id, path, old, new):
    """A FileReview for an existing file, built the same way Agent does it."""
    return FileReview(call_id=call_id, path=path, is_new=not old, hunks=split_hunks(old, new))


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


# -- review_batch: the same y/n/a/d flow, but spanning every file in a round --


def test_review_batch_empty_returns_empty_dict():
    assert make_ui().review_batch([]) == {}


def test_review_batch_independent_choices_across_files(monkeypatch):
    a = make_review("a", "a.txt", "a\nb\nc\nd\n", "A\nb\nC\nd\n")
    b = make_review("b", "b.txt", "p\nq\nr\ns\n", "P\nq\nR\ns\n")
    assert len(a.hunks) == 2 and len(b.hunks) == 2

    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["y", "n", "n", "y"]))
    decisions = ui.review_batch([a, b])
    assert decisions == {"a": [True, False], "b": [False, True]}


def test_review_batch_accept_all_spans_every_remaining_file(monkeypatch):
    a = make_review("a", "a.txt", "a\nb\nc\nd\n", "A\nb\nC\nd\n")
    b = make_review("b", "b.txt", "p\nq\nr\ns\n", "P\nq\nR\ns\n")

    ui = make_ui()
    # "a" on the very first hunk of the first file should also apply every
    # remaining hunk of every remaining file, without prompting again.
    monkeypatch.setattr(ui.console, "input", scripted_input(["a"]))
    decisions = ui.review_batch([a, b])
    assert decisions == {"a": [True, True], "b": [True, True]}


def test_review_batch_decline_all_spans_every_remaining_file(monkeypatch):
    a = make_review("a", "a.txt", "a\nb\nc\nd\n", "A\nb\nC\nd\n")
    b = make_review("b", "b.txt", "p\nq\nr\ns\n", "P\nq\nR\ns\n")

    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["d"]))
    decisions = ui.review_batch([a, b])
    assert decisions == {"a": [False, False], "b": [False, False]}


def test_review_batch_new_file_gets_one_whole_file_prompt(monkeypatch):
    new_file = FileReview(call_id="n", path="new.txt", is_new=True, hunks=[], new_content="hello\n")
    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["y"]))
    assert ui.review_batch([new_file]) == {"n": [True]}


def test_review_batch_new_file_declined(monkeypatch):
    new_file = FileReview(call_id="n", path="new.txt", is_new=True, hunks=[], new_content="hello\n")
    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["n"]))
    assert ui.review_batch([new_file]) == {"n": [False]}


def test_review_batch_accept_all_spans_hunks_and_new_files_together(monkeypatch):
    existing = make_review("e", "a.txt", "a\nb\nc\nd\n", "A\nb\nC\nd\n")
    new_file = FileReview(call_id="n", path="new.txt", is_new=True, hunks=[], new_content="hello\n")

    ui = make_ui()
    monkeypatch.setattr(ui.console, "input", scripted_input(["a"]))
    decisions = ui.review_batch([existing, new_file])
    assert decisions == {"e": [True, True], "n": [True]}


def test_review_batch_shows_summary_only_when_more_than_one_file(monkeypatch):
    a = make_review("a", "a.txt", "a\nb\n", "A\nb\n")
    b = make_review("b", "b.txt", "p\nq\n", "P\nq\n")

    ui = UI(Console(record=True, force_terminal=False, width=200))
    monkeypatch.setattr(ui.console, "input", scripted_input(["a"]))
    ui.review_batch([a, b])
    assert "2 file(s) will change" in ui.console.export_text()

    ui2 = UI(Console(record=True, force_terminal=False, width=200))
    monkeypatch.setattr(ui2.console, "input", scripted_input(["y"]))
    ui2.review_batch([a])
    assert "file(s) will change" not in ui2.console.export_text()
