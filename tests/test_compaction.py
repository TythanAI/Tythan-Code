"""Pure-function tests for the compaction helpers (round splitting, heuristics)."""

from tythancode.compaction import (
    cap_head,
    estimate_tokens_heuristic,
    is_round_boundary,
    split_into_rounds,
)


def test_is_round_boundary_true_for_plain_user_text():
    assert is_round_boundary({"role": "user", "content": "hello"})


def test_is_round_boundary_false_for_tool_result_shapes():
    # Anthropic tool_result: role "user" but list content, not a plain string.
    assert not is_round_boundary(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}
    )
    # OpenAI tool result: role "tool".
    assert not is_round_boundary({"role": "tool", "tool_call_id": "x", "content": "y"})
    # Assistant messages are never boundaries.
    assert not is_round_boundary({"role": "assistant", "content": "hi"})


def test_split_into_rounds_groups_by_user_boundary():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},  # not a boundary
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 3"},
    ]
    rounds = split_into_rounds(messages)
    assert len(rounds) == 2
    assert [m["content"] for m in rounds[0]] == [
        "first",
        "reply 1",
        [{"type": "tool_result", "content": "x"}],
        "reply 2",
    ]
    assert [m["content"] for m in rounds[1]] == ["second", "reply 3"]


def test_split_into_rounds_empty():
    assert split_into_rounds([]) == []


def test_split_into_rounds_leading_non_boundary_kept():
    # Shouldn't happen in practice (conversations start with a user message),
    # but must not silently drop messages if it does.
    messages = [{"role": "tool", "tool_call_id": "x", "content": "orphan"}]
    rounds = split_into_rounds(messages)
    assert rounds == [messages]


def test_estimate_tokens_heuristic_grows_with_content():
    small = estimate_tokens_heuristic([{"role": "user", "content": "hi"}])
    large = estimate_tokens_heuristic([{"role": "user", "content": "hi " * 1000}])
    assert large > small
    assert small >= 0


def test_estimate_tokens_heuristic_includes_system_prompt():
    messages = [{"role": "user", "content": "hi"}]
    without_system = estimate_tokens_heuristic(messages, "")
    with_system = estimate_tokens_heuristic(messages, "x" * 4000)
    assert with_system > without_system


def test_estimate_tokens_heuristic_handles_non_serializable_content():
    class Weird:
        def __str__(self):
            return "weird-object"

    messages = [{"role": "assistant", "content": [Weird()]}]
    # Must not raise — json.dumps(default=str) should stringify it.
    assert estimate_tokens_heuristic(messages) > 0


def test_cap_head_short_text_unchanged():
    assert cap_head("short", 100) == "short"


def test_cap_head_truncates_from_the_front_keeping_the_tail():
    text = "0123456789" * 10  # 100 chars
    capped = cap_head(text, 20)
    assert capped.endswith(text[-20:])
    assert "[earlier content omitted]" in capped
    assert len(capped) < len(text)
