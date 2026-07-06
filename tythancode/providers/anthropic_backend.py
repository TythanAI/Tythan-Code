"""Native Anthropic Messages API backend (streaming, adaptive thinking, caching)."""

from __future__ import annotations

import json

import anthropic

from ..config import Config, ProviderConfig, default_context_window
from ..tools import truncate
from ..ui import UI
from .base import Backend, ToolCall, ToolResult, TurnResult


def _format_usage(usage) -> str:
    parts = [f"{usage.input_tokens} in", f"{usage.output_tokens} out"]
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    if cached:
        parts.append(f"{cached} cached")
    return " / ".join(parts)


def _total_context_tokens(usage) -> int:
    """Best estimate of how many tokens the *whole* sent context used up,
    combining freshly-processed, cache-write and cache-read tokens."""
    return (
        (getattr(usage, "input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )


class AnthropicBackend(Backend):
    def __init__(self, pcfg: ProviderConfig, config: Config, client: anthropic.Anthropic | None = None):
        super().__init__(pcfg.model, context_window=default_context_window(pcfg))
        self.name = pcfg.name
        self.config = config  # effort/max_tokens are read live so /effort applies
        kwargs = {}
        if pcfg.base_url:
            kwargs["base_url"] = pcfg.base_url
        self.client = client or anthropic.Anthropic(**kwargs)

    def add_user_message(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.output,
                        "is_error": r.is_error,
                    }
                    for r in results
                ],
            }
        )

    def stream_turn(self, messages: list, system: str, tools: list[dict], ui: UI) -> TurnResult:
        while True:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.config.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={"effort": self.config.effort},
                tools=tools,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "thinking":
                            ui.thinking_started()
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            ui.stream_text(event.delta.text)
                ui.flush_stream()
                response = stream.get_final_message()

            if response.usage is not None:
                self.last_context_tokens = _total_context_tokens(response.usage)

            if response.stop_reason == "refusal":
                # Discard the (empty or partial) refused output; keep prior history.
                return TurnResult("refusal")

            # Keep full content (incl. thinking/tool_use blocks) in history.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue  # server-side pause; re-send to resume

            calls = [
                ToolCall(id=b.id, name=b.name, input=b.input)
                for b in response.content
                if b.type == "tool_use"
            ]
            usage = _format_usage(response.usage)
            if calls:
                return TurnResult("tool_use", calls, usage)
            if response.stop_reason == "max_tokens":
                return TurnResult("length", [], usage)
            return TurnResult("end", [], usage)

    # -- context-compaction support ---------------------------------------

    def render_round(self, round_messages: list) -> str:
        lines: list[str] = []
        for m in round_messages:
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
                continue
            for block in content or []:
                btype = getattr(block, "type", None)
                if btype is None and isinstance(block, dict):
                    btype = block.get("type")
                if btype == "text":
                    text = getattr(block, "text", None)
                    if text is None and isinstance(block, dict):
                        text = block.get("text", "")
                    lines.append(f"{role}: {text}")
                elif btype == "tool_use":
                    name = getattr(block, "name", None)
                    if name is None and isinstance(block, dict):
                        name = block.get("name")
                    inp = getattr(block, "input", None)
                    if inp is None and isinstance(block, dict):
                        inp = block.get("input")
                    args = truncate(json.dumps(inp, default=str, ensure_ascii=False), 500)
                    lines.append(f"{role} called tool {name}({args})")
                elif btype == "tool_result":
                    out = block.get("content") if isinstance(block, dict) else getattr(block, "content", "")
                    text = out if isinstance(out, str) else json.dumps(out, default=str, ensure_ascii=False)
                    lines.append(f"tool result: {truncate(text, 1000)}")
                # "thinking" blocks are internal reasoning — skip them in the summary transcript.
        return "\n".join(lines)

    def complete_text(self, system: str, user_text: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
