"""Agent-level tests for context compaction and checkpoint/undo wiring.

Uses a scripted FakeBackend (no network), same style as test_agent.py, but one
that implements render_round/complete_text/context_window so compaction can
actually be exercised. Checkpoint storage is always pointed at tmp_path so
these tests never touch the real ~/.tythancode directory.
"""

from rich.console import Console

from tythancode.agent import Agent
from tythancode.checkpoints import CheckpointStore
from tythancode.config import Config
from tythancode.providers.base import Backend, ToolCall, TurnResult
from tythancode.ui import UI


class ScriptedBackend(Backend):
    """Backend whose stream_turn results are pre-scripted, and whose
    compaction hooks are simple, inspectable fakes."""

    name = "scripted"

    def __init__(self, turns=None, context_window=32_000, summary="SUMMARY"):
        super().__init__(model="fake-model", context_window=context_window)
        self.turns = list(turns or [])
        self.calls = 0
        self.summary_calls: list[str] = []
        self.summary_text = summary
        self.complete_text_error: Exception | None = None

    def add_user_message(self, messages, text):
        messages.append({"role": "user", "content": text})

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool_results", "results": results})

    def stream_turn(self, messages, system, tools, ui):
        self.calls += 1
        if self.turns:
            return self.turns.pop(0)
        return TurnResult("end")

    def complete_text(self, system, user_text):
        if self.complete_text_error:
            raise self.complete_text_error
        self.summary_calls.append(user_text)
        return self.summary_text


def make_agent(tmp_path, backend, **config_kwargs):
    config = Config(workspace=tmp_path, yolo=True, **config_kwargs)
    ui = UI(Console(file=open("/dev/null", "w"), force_terminal=False))
    store = CheckpointStore(tmp_path, storage_dir=tmp_path / ".checkpoints")
    return Agent(config, ui, backend, checkpoint_store=store,
                 code_index_cache_dir=tmp_path / ".index-cache")


# -- checkpoints --------------------------------------------------------


def test_write_file_creates_undoable_checkpoint(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "v1"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.run_turn("create a.txt")

    assert (tmp_path / "a.txt").read_text() == "v1"
    cps = agent.checkpoints.list()
    assert len(cps) == 1
    assert cps[0].changes[0].existed_before is False

    agent.checkpoints.undo_last()
    assert not (tmp_path / "a.txt").exists()


def test_edit_file_checkpoint_restores_previous_content(tmp_path):
    (tmp_path / "a.txt").write_text("original")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="edit_file",
                                          input={"path": "a.txt", "old_string": "original",
                                                 "new_string": "changed"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.run_turn("change a.txt")

    assert (tmp_path / "a.txt").read_text() == "changed"
    agent.checkpoints.undo_last()
    assert (tmp_path / "a.txt").read_text() == "original"


def test_declined_write_creates_no_checkpoint(tmp_path, monkeypatch):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "v1"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.config.yolo = False
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: False)
    monkeypatch.setattr(agent.ui, "show_diff", lambda *a, **k: None)

    agent.run_turn("create a.txt")

    assert not (tmp_path / "a.txt").exists()
    assert agent.checkpoints.list() == []


def test_edit_file_full_hunk_acceptance(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("line1\nline2\nline3\n")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="edit_file",
                                          input={"path": "a.txt", "old_string": "line2", "new_string": "CHANGED"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.config.yolo = False
    monkeypatch.setattr(agent.ui, "review_hunks", lambda path, hunks: [True] * len(hunks))

    agent.run_turn("change line2")

    assert (tmp_path / "a.txt").read_text() == "line1\nCHANGED\nline3\n"
    assert len(agent.checkpoints.list()) == 1
    result = agent.messages[1]["results"][0]
    assert result.is_error is False
    assert "hunk(s) applied" not in result.output  # fully applied — no partial-note noise


def test_edit_file_full_hunk_rejection_creates_no_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("line1\nline2\nline3\n")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="edit_file",
                                          input={"path": "a.txt", "old_string": "line2", "new_string": "CHANGED"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.config.yolo = False
    monkeypatch.setattr(agent.ui, "review_hunks", lambda path, hunks: [False] * len(hunks))

    agent.run_turn("change line2")

    assert (tmp_path / "a.txt").read_text() == "line1\nline2\nline3\n"
    assert agent.checkpoints.list() == []
    result = agent.messages[1]["results"][0]
    assert result.is_error is True
    assert "declined" in result.output


def test_write_file_partial_hunk_acceptance(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a\nb\nc\nd\ne\n")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "A\nb\nc\nD\ne\n"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.config.yolo = False
    captured_hunks = []

    def fake_review(path, hunks):
        captured_hunks.extend(hunks)
        return [False, True]  # reject a->A, accept d->D

    monkeypatch.setattr(agent.ui, "review_hunks", fake_review)

    agent.run_turn("update a.txt")

    assert len(captured_hunks) == 2  # write_file to an existing file also gets per-hunk review
    assert (tmp_path / "a.txt").read_text() == "a\nb\nc\nD\ne\n"
    result = agent.messages[1]["results"][0]
    assert result.is_error is False
    assert "1/2 hunk(s) applied" in result.output
    assert len(agent.checkpoints.list()) == 1


def test_write_file_to_new_file_skips_hunk_review(tmp_path, monkeypatch):
    """A brand-new file has nothing to diff against, so it still gets the
    simple whole-file confirm rather than hunk review."""
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "new.txt", "content": "hello\n"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.config.yolo = False
    monkeypatch.setattr(agent.ui, "review_hunks", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a brand-new file should not go through hunk review")))
    monkeypatch.setattr(agent.ui, "show_diff", lambda *a, **k: None)
    monkeypatch.setattr(agent.ui, "confirm", lambda prompt: True)

    agent.run_turn("create new.txt")
    assert (tmp_path / "new.txt").read_text() == "hello\n"


def test_read_only_turn_creates_no_checkpoint(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="read_file", input={"path": "hello.txt"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.run_turn("what's in hello.txt?")
    assert agent.checkpoints.list() == []


def test_checkpoints_disabled_via_config(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "v1"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns), checkpoints_enabled=False)
    agent.run_turn("create a.txt")

    assert (tmp_path / "a.txt").read_text() == "v1"
    assert agent.checkpoints.list() == []


def test_multi_file_turn_is_one_checkpoint(tmp_path):
    turns = [
        TurnResult("tool_use", [
            ToolCall(id="tu_1", name="write_file", input={"path": "a.txt", "content": "a"}),
            ToolCall(id="tu_2", name="write_file", input={"path": "b.txt", "content": "b"}),
        ]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    agent.run_turn("create both files")

    cps = agent.checkpoints.list()
    assert len(cps) == 1
    assert {c.path for c in cps[0].changes} == {str(tmp_path / "a.txt"), str(tmp_path / "b.txt")}

    agent.checkpoints.undo_last()
    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


# -- compaction -----------------------------------------------------------


def test_maybe_compact_noop_when_history_is_small(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([]))
    agent.messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert agent.maybe_compact() is False
    assert len(agent.messages) == 2


def test_maybe_compact_forced_summarizes_old_rounds_keeps_recent(tmp_path):
    backend = ScriptedBackend([], summary="the user asked X, we did Y")
    agent = make_agent(tmp_path, backend, compact_keep_rounds=1)
    for i in range(4):
        agent.messages.append({"role": "user", "content": f"turn {i}"})
        agent.messages.append({"role": "assistant", "content": f"reply {i}"})

    compacted = agent.maybe_compact(force=True)

    assert compacted is True
    assert len(backend.summary_calls) == 1
    # only the last round (1 kept) should remain, prefixed with the summary
    assert len(agent.messages) == 2
    assert "the user asked X, we did Y" in agent.messages[0]["content"]
    assert "turn 3" in agent.messages[0]["content"]
    assert agent.messages[1]["content"] == "reply 3"


def test_maybe_compact_never_summarizes_away_the_newest_round(tmp_path):
    backend = ScriptedBackend([], summary="summary")
    agent = make_agent(tmp_path, backend, compact_keep_rounds=2)
    for i in range(3):
        agent.messages.append({"role": "user", "content": f"turn {i}"})
        agent.messages.append({"role": "assistant", "content": f"reply {i}"})

    agent.maybe_compact(force=True)
    # last message is still the newest assistant reply, never dropped
    assert agent.messages[-1]["content"] == "reply 2"


def test_maybe_compact_not_enough_rounds_is_a_noop_even_forced(tmp_path):
    backend = ScriptedBackend([])
    agent = make_agent(tmp_path, backend, compact_keep_rounds=5)
    agent.messages = [
        {"role": "user", "content": "only one round"},
        {"role": "assistant", "content": "reply"},
    ]
    assert agent.maybe_compact(force=True) is False
    assert len(backend.summary_calls) == 0


def test_maybe_compact_auto_triggers_when_over_budget(tmp_path):
    # Tiny context window makes the token budget tiny, so even a short
    # history should cross the compaction threshold automatically.
    backend = ScriptedBackend([], context_window=1200, summary="compacted")
    agent = make_agent(tmp_path, backend, max_tokens=100, compact_keep_rounds=1)
    for i in range(4):
        agent.messages.append({"role": "user", "content": f"turn {i} " + "x" * 2000})
        agent.messages.append({"role": "assistant", "content": f"reply {i}"})

    assert agent.maybe_compact() is True
    assert len(backend.summary_calls) == 1


def test_maybe_compact_uses_real_backend_usage_over_heuristic(tmp_path):
    backend = ScriptedBackend([], context_window=1_000_000, summary="compacted")
    agent = make_agent(tmp_path, backend, compact_keep_rounds=1)
    for i in range(3):  # more rounds than compact_keep_rounds, so there's something to summarize
        agent.messages.append({"role": "user", "content": f"tiny {i}"})
        agent.messages.append({"role": "assistant", "content": f"ok {i}"})
    backend.last_context_tokens = 999_999  # way over budget despite the tiny history
    assert agent.context_tokens_estimate() == 999_999
    assert agent.maybe_compact() is True  # would be False on the heuristic alone


def test_compaction_failure_is_reported_and_does_not_crash(tmp_path):
    backend = ScriptedBackend([])
    backend.complete_text_error = ConnectionError("network down")
    agent = make_agent(tmp_path, backend, compact_keep_rounds=1)
    for i in range(3):
        agent.messages.append({"role": "user", "content": f"turn {i}"})
        agent.messages.append({"role": "assistant", "content": f"reply {i}"})
    before = list(agent.messages)

    result = agent.maybe_compact(force=True)

    assert result is False
    assert agent.messages == before  # history untouched on failure
    assert agent._compaction_unavailable is True


def test_compaction_unavailable_flag_skips_further_attempts_this_turn(tmp_path):
    backend = ScriptedBackend(
        [TurnResult("end"), TurnResult("end")],
        context_window=1200,
    )
    backend.complete_text_error = RuntimeError("boom")
    agent = make_agent(tmp_path, backend, max_tokens=100, compact_keep_rounds=1)
    for i in range(4):
        agent.messages.append({"role": "user", "content": f"turn {i} " + "x" * 2000})
        agent.messages.append({"role": "assistant", "content": f"reply {i}"})

    # First explicit attempt fails and flips the flag.
    assert agent.maybe_compact() is False
    assert agent._compaction_unavailable is True
    # A second attempt in the same turn should short-circuit without calling
    # complete_text again (no exception raised means it didn't try).
    assert agent.maybe_compact() is False


def test_run_turn_resets_compaction_unavailable_flag_each_turn(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([TurnResult("end")]))
    agent._compaction_unavailable = True
    agent.run_turn("hello")
    assert agent._compaction_unavailable is False


# -- token_budget ---------------------------------------------------------


def test_token_budget_caps_reserve_for_small_local_context_windows(tmp_path):
    """A small context_window (e.g. the 8k local-model default) must not have
    the full default max_tokens (64k) reserved out of it — that would leave
    almost nothing for actual conversation and trigger compaction constantly."""
    backend = ScriptedBackend([], context_window=8_000)
    agent = make_agent(tmp_path, backend)  # default max_tokens is 64_000
    # reserve = min(64_000, 8_000 // 2) = 4_000 -> budget = 8_000 - 4_000 = 4_000
    assert agent.token_budget() == 4_000


def test_token_budget_unaffected_for_large_context_windows(tmp_path):
    """For a large window where max_tokens is already well under half of it,
    the cap must not change anything (no regression for the common/default case)."""
    backend = ScriptedBackend([], context_window=200_000)
    agent = make_agent(tmp_path, backend)  # max_tokens defaults to 64_000
    assert agent.token_budget() == 200_000 - 64_000


# -- checkpoint label -------------------------------------------------------


def test_run_turn_uses_explicit_label_for_checkpoint_over_user_input(tmp_path):
    turns = [
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "v1"})]),
        TurnResult("end"),
    ]
    agent = make_agent(tmp_path, ScriptedBackend(turns))
    expanded = "@notes.md summarize this\n\n<file path=\"notes.md\">huge file contents...</file>"
    agent.run_turn(expanded, label="@notes.md summarize this")

    cps = agent.checkpoints.list()
    assert len(cps) == 1
    assert cps[0].label == "@notes.md summarize this"
    # The model still sees the full expanded text as its actual input.
    assert agent.messages[0]["content"] == expanded


def test_run_turn_label_defaults_to_user_input(tmp_path):
    agent = make_agent(tmp_path, ScriptedBackend([
        TurnResult("tool_use", [ToolCall(id="tu_1", name="write_file",
                                          input={"path": "a.txt", "content": "v1"})]),
        TurnResult("end"),
    ]))
    agent.run_turn("create a.txt")
    assert agent.checkpoints.list()[0].label == "create a.txt"


# -- finally-block resilience ----------------------------------------------


def test_run_turn_survives_commit_turn_raising_oserror(tmp_path, monkeypatch):
    """A disk error while saving the checkpoint must not crash run_turn, and
    must not replace/hide a real exception already propagating from the turn."""
    agent = make_agent(tmp_path, ScriptedBackend([TurnResult("end")]))

    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(agent.checkpoints, "commit_turn", boom)
    agent.run_turn("hello")  # must not raise


def test_run_turn_commit_turn_oserror_does_not_mask_real_exception(tmp_path, monkeypatch):
    class ExplodingBackend(ScriptedBackend):
        def stream_turn(self, messages, system, tools, ui):
            raise ConnectionError("network down")

    agent = make_agent(tmp_path, ExplodingBackend([]))
    monkeypatch.setattr(agent.checkpoints, "commit_turn", lambda: (_ for _ in ()).throw(OSError("disk full")))

    try:
        agent.run_turn("hello")
        assert False, "expected ConnectionError to propagate"
    except ConnectionError:
        pass  # the *original* exception must win, not the finally-block's OSError
