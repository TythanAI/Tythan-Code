"""Backend for any OpenAI-compatible chat-completions endpoint.

Covers OpenAI itself plus OpenRouter, Groq, DeepSeek, Mistral, xAI, and local
servers (Ollama, LM Studio, vLLM) — anything speaking /v1/chat/completions
with function tools and streaming.
"""

from __future__ import annotations

import json
import os

from openai import BadRequestError, UnprocessableEntityError

from ..config import LOCAL_HOSTS, ProviderConfig, default_context_window
from ..tools import truncate
from ..ui import UI
from .base import Backend, BackendConfigError, ToolCall, ToolResult, TurnResult


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _resolve_api_key(pcfg: ProviderConfig) -> str:
    if pcfg.api_key_env:
        key = os.environ.get(pcfg.api_key_env)
        if key:
            return key
    if pcfg.base_url and any(h in pcfg.base_url for h in LOCAL_HOSTS):
        return "local"  # local servers don't check the key
    hint = pcfg.api_key_env or "an api_key_env entry in ~/.tythancode/config.json"
    raise BackendConfigError(f"provider '{pcfg.name}' needs an API key — set {hint}")


class OpenAIBackend(Backend):
    def __init__(self, pcfg: ProviderConfig, client=None):
        super().__init__(pcfg.model, context_window=default_context_window(pcfg))
        self.name = pcfg.name
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=pcfg.base_url, api_key=_resolve_api_key(pcfg))
        # None = untested, True/False = known from a prior call. Some
        # OpenAI-compatible endpoints (notably some local servers) reject the
        # stream_options parameter outright, so we probe once and remember.
        self._usage_supported: bool | None = None

    def add_user_message(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        for r in results:
            messages.append(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.output}
            )

    def _create_stream(self, full_messages: list, tools: list[dict]):
        kwargs = dict(model=self.model, messages=full_messages, tools=to_openai_tools(tools), stream=True)
        if self._usage_supported is False:
            return self.client.chat.completions.create(**kwargs)
        try:
            stream = self.client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
            self._usage_supported = True
            return stream
        except (BadRequestError, UnprocessableEntityError, TypeError):
            # These specifically mean "the request body/params were rejected" —
            # the closest thing to a reliable signal that stream_options isn't
            # recognized. Remember it and fall back so we don't pay for a
            # failed request every call. Anything else (auth, rate limit,
            # network, 5xx) is unrelated to stream_options support and must
            # propagate normally instead of being silently misclassified and
            # permanently disabling usage tracking for an unrelated, possibly
            # transient reason.
            self._usage_supported = False
            return self.client.chat.completions.create(**kwargs)

    def stream_turn(self, messages: list, system: str, tools: list[dict], ui: UI) -> TurnResult:
        stream = self._create_stream([{"role": "system", "content": system}] + messages, tools)

        text_parts: list[str] = []
        pending: dict[int, dict] = {}  # tool-call accumulator keyed by index
        finish_reason = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                if prompt_tokens is not None:
                    self.last_context_tokens = prompt_tokens
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta and delta.content:
                text_parts.append(delta.content)
                ui.stream_text(delta.content)
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["args"] += tc.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        ui.flush_stream()

        assistant: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        calls: list[ToolCall] = []
        if pending:
            assistant["tool_calls"] = []
            for i, slot in sorted(pending.items()):
                call_id = slot["id"] or f"call_{i}"
                assistant["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot["args"]},
                    }
                )
                try:
                    parsed = json.loads(slot["args"]) if slot["args"].strip() else {}
                except json.JSONDecodeError:
                    parsed = {}
                calls.append(ToolCall(id=call_id, name=slot["name"], input=parsed))
        messages.append(assistant)

        if calls:
            return TurnResult("tool_use", calls)
        if finish_reason == "length":
            return TurnResult("length")
        return TurnResult("end")

    # -- context-compaction support ---------------------------------------

    def render_round(self, round_messages: list) -> str:
        lines: list[str] = []
        for m in round_messages:
            role = m.get("role", "?")
            if role == "tool":
                lines.append(f"tool result: {truncate(str(m.get('content', '')), 1000)}")
                continue
            content = m.get("content")
            if isinstance(content, str) and content:
                lines.append(f"{role}: {content}")
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name")
                args = truncate(str(fn.get("arguments", "")), 500)
                lines.append(f"{role} called tool {name}({args})")
        return "\n".join(lines)

    def complete_text(self, system: str, user_text: str) -> str:
        # Capped like AnthropicBackend.complete_text — this is only ever used
        # for a compaction summary, so a verbose/non-compliant model can't
        # return an unbounded completion that defeats the point of compacting.
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        )
        return response.choices[0].message.content or ""
