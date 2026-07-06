"""Tests for the compaction/checkpoint-supporting hooks added to each backend:
context_window defaults, usage capture, render_round, complete_text. All
offline — network clients are replaced with simple fakes."""

import os
from types import SimpleNamespace

import pytest
from rich.console import Console

from tythancode.config import (
    DEFAULT_ANTHROPIC_CONTEXT_WINDOW,
    DEFAULT_GENERIC_CONTEXT_WINDOW,
    DEFAULT_LOCAL_CONTEXT_WINDOW,
    Config,
    ProviderConfig,
    default_context_window,
)
from tythancode.providers.anthropic_backend import AnthropicBackend, _total_context_tokens
from tythancode.providers.openai_backend import OpenAIBackend
from tythancode.ui import UI


def make_ui() -> UI:
    return UI(Console(file=open(os.devnull, "w"), force_terminal=False))


# -- default_context_window ------------------------------------------------


def test_default_context_window_explicit_override_wins():
    pcfg = ProviderConfig(name="x", type="openai", model="m", context_window=9999)
    assert default_context_window(pcfg) == 9999


def test_default_context_window_anthropic():
    pcfg = ProviderConfig(name="anthropic", type="anthropic", model="m")
    assert default_context_window(pcfg) == DEFAULT_ANTHROPIC_CONTEXT_WINDOW


def test_default_context_window_local_host_is_conservative():
    pcfg = ProviderConfig(name="ollama", type="openai", model="m", base_url="http://localhost:11434/v1")
    assert default_context_window(pcfg) == DEFAULT_LOCAL_CONTEXT_WINDOW


def test_default_context_window_known_hosted_provider():
    pcfg = ProviderConfig(name="openai", type="openai", model="m", base_url="https://api.openai.com/v1")
    assert default_context_window(pcfg) == 128_000


def test_default_context_window_unknown_host_falls_back_to_generic():
    pcfg = ProviderConfig(name="mystery", type="openai", model="m", base_url="https://example.com/v1")
    assert default_context_window(pcfg) == DEFAULT_GENERIC_CONTEXT_WINDOW


# -- Anthropic backend -------------------------------------------------------


def make_anthropic_backend(tmp_path, client=None) -> AnthropicBackend:
    pcfg = ProviderConfig(name="anthropic", type="anthropic", model="claude-x")
    config = Config(workspace=tmp_path)
    return AnthropicBackend(pcfg, config, client=client or SimpleNamespace())


def test_anthropic_backend_context_window_default(tmp_path):
    backend = make_anthropic_backend(tmp_path)
    assert backend.context_window == DEFAULT_ANTHROPIC_CONTEXT_WINDOW


def test_total_context_tokens_sums_all_usage_kinds():
    usage = SimpleNamespace(input_tokens=100, cache_read_input_tokens=50, cache_creation_input_tokens=25)
    assert _total_context_tokens(usage) == 175


def test_total_context_tokens_handles_missing_fields():
    usage = SimpleNamespace(input_tokens=100)  # no cache_* attributes at all
    assert _total_context_tokens(usage) == 100


def test_anthropic_complete_text(tmp_path):
    class Block:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    def create(**kwargs):
        assert kwargs["system"] == "summarize this"
        return SimpleNamespace(content=[Block("text", "the summary")])

    backend = make_anthropic_backend(tmp_path, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert backend.complete_text("summarize this", "old transcript") == "the summary"


def test_anthropic_complete_text_ignores_non_text_blocks(tmp_path):
    class Block:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    def create(**kwargs):
        return SimpleNamespace(content=[Block("thinking", "internal"), Block("text", "visible")])

    backend = make_anthropic_backend(tmp_path, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert backend.complete_text("sys", "text") == "visible"


def test_anthropic_render_round_mixes_text_tool_use_and_tool_result(tmp_path):
    class Block:
        """Mimics an Anthropic SDK content-block object (attribute access, not dict)."""

        def __init__(self, type_, **kw):
            self.type = type_
            for k, v in kw.items():
                setattr(self, k, v)

    backend = make_anthropic_backend(tmp_path)
    round_messages = [
        {"role": "user", "content": "please fix the bug"},
        {
            "role": "assistant",
            "content": [
                Block("text", text="Let me look"),
                Block("tool_use", name="read_file", input={"path": "a.py"}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents", "is_error": False}
            ],
        },
    ]
    text = backend.render_round(round_messages)
    assert "user: please fix the bug" in text
    assert "Let me look" in text
    assert "called tool read_file" in text
    assert '"path": "a.py"' in text
    assert "tool result: file contents" in text


def test_anthropic_render_round_skips_thinking_blocks(tmp_path):
    class Block:
        def __init__(self, type_, **kw):
            self.type = type_
            for k, v in kw.items():
                setattr(self, k, v)

    backend = make_anthropic_backend(tmp_path)
    round_messages = [
        {"role": "assistant", "content": [Block("thinking", thinking="secret reasoning"), Block("text", text="final answer")]},
    ]
    text = backend.render_round(round_messages)
    assert "secret reasoning" not in text
    assert "final answer" in text


# -- OpenAI-compatible backend ------------------------------------------------


def make_fake_openai_client(create_fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def make_openai_backend(client) -> OpenAIBackend:
    pcfg = ProviderConfig(name="local", type="openai", model="m", base_url="http://localhost:11434/v1")
    return OpenAIBackend(pcfg, client=client)


def test_openai_backend_context_window_default_for_local_host():
    backend = make_openai_backend(SimpleNamespace())
    assert backend.context_window == DEFAULT_LOCAL_CONTEXT_WINDOW


def test_create_stream_uses_stream_options_when_supported():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return iter([])

    backend = make_openai_backend(make_fake_openai_client(create))
    backend._create_stream([{"role": "user", "content": "hi"}], [])

    assert backend._usage_supported is True
    assert len(calls) == 1
    assert calls[0]["stream_options"] == {"include_usage": True}


def test_create_stream_falls_back_and_remembers_when_unsupported():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "stream_options" in kwargs:
            # TypeError is one of the narrow set of "the request itself was
            # rejected" signals _create_stream treats as "unsupported param".
            raise TypeError("unexpected keyword argument 'stream_options'")
        return iter([])

    backend = make_openai_backend(make_fake_openai_client(create))
    backend._create_stream([{"role": "user", "content": "hi"}], [])

    assert backend._usage_supported is False
    assert len(calls) == 2
    assert "stream_options" in calls[0]
    assert "stream_options" not in calls[1]

    # Next call shouldn't retry stream_options at all — remembered as unsupported.
    backend._create_stream([{"role": "user", "content": "hi"}], [])
    assert len(calls) == 3


def test_create_stream_does_not_misclassify_unrelated_failures():
    """An auth/rate-limit/network/server error on the stream_options probe is
    NOT the same thing as 'this endpoint doesn't support stream_options' — it
    must propagate immediately instead of silently and permanently disabling
    usage tracking for an unrelated reason."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("network blip")

    backend = make_openai_backend(make_fake_openai_client(create))

    with pytest.raises(ConnectionError):
        backend._create_stream([{"role": "user", "content": "hi"}], [])

    # Not misclassified as "unsupported": no silent fallback call, and the
    # probe stays untested so a later, working call can still succeed with
    # stream_options.
    assert len(calls) == 1
    assert backend._usage_supported is None


def test_stream_turn_captures_prompt_tokens_from_usage_chunk():
    class Delta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class Choice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class Chunk:
        def __init__(self, choices=None, usage=None):
            self.choices = choices or []
            self.usage = usage

    chunks = [
        Chunk(choices=[Choice(Delta(content="hi"))]),
        Chunk(choices=[Choice(Delta(content=None), finish_reason="stop")]),
        Chunk(choices=[], usage=SimpleNamespace(prompt_tokens=4321)),
    ]

    backend = make_openai_backend(make_fake_openai_client(lambda **kwargs: iter(chunks)))
    result = backend.stream_turn([], "system prompt", [], make_ui())

    assert backend.last_context_tokens == 4321
    assert result.stop == "end"


def test_openai_complete_text():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="a summary"))]
        )

    backend = make_openai_backend(make_fake_openai_client(create))
    assert backend.complete_text("sys", "old transcript") == "a summary"
    # Capped like AnthropicBackend.complete_text so a verbose local model can't
    # return an unbounded summary that defeats the point of compacting.
    assert captured["max_tokens"] == 2000


def test_openai_render_round_includes_tool_calls_and_tool_results():
    backend = make_openai_backend(SimpleNamespace())
    round_messages = [
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        {"role": "assistant", "content": "fixed it"},
    ]
    text = backend.render_round(round_messages)
    assert "user: fix the bug" in text
    assert "called tool read_file" in text
    assert "tool result: file contents" in text
    assert "fixed it" in text
