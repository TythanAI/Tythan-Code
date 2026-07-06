"""Backend interface shared by all providers.

A Backend owns the wire format of the conversation history: the agent keeps a
plain list and only manipulates it through the backend, so each provider can
store messages in its native shape (Anthropic content blocks vs OpenAI chat
messages). Switching providers therefore resets the conversation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import DEFAULT_GENERIC_CONTEXT_WINDOW
from ..tools import truncate
from ..ui import UI


class BackendConfigError(Exception):
    """Provider misconfiguration (missing key, bad endpoint, ...)."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False


@dataclass
class TurnResult:
    stop: str  # "tool_use" | "end" | "refusal" | "length"
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: str = ""  # human-readable token usage, may be empty


class Backend(ABC):
    name: str = "base"

    def __init__(self, model: str, context_window: int = DEFAULT_GENERIC_CONTEXT_WINDOW):
        self.model = model
        self.context_window = context_window
        # Real input-token count for the history as of the last stream_turn call,
        # when the provider reported one. None until a call has completed, or if
        # the provider never reports usage — callers fall back to a heuristic.
        self.last_context_tokens: int | None = None

    def describe(self) -> str:
        return f"{self.name} / {self.model}"

    @abstractmethod
    def add_user_message(self, messages: list, text: str) -> None: ...

    @abstractmethod
    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def stream_turn(self, messages: list, system: str, tools: list[dict], ui: UI) -> TurnResult:
        """Make one model call: stream text to the UI, append the assistant
        message(s) to `messages` in native format, and report tool calls."""

    # -- context-compaction support ---------------------------------------

    def render_round(self, round_messages: list) -> str:
        """Best-effort plain-text rendering of one round of native messages,
        used to build the transcript fed to `complete_text` when summarizing
        old history. Subclasses override this for a nicer, format-aware
        rendering; this generic fallback just stringifies everything."""
        lines = []
        for m in round_messages:
            role = m.get("role", "?") if isinstance(m, dict) else "?"
            content = m.get("content") if isinstance(m, dict) else m
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
            else:
                lines.append(f"{role}: {truncate(json.dumps(content, default=str, ensure_ascii=False), 2000)}")
        return "\n".join(lines)

    def complete_text(self, system: str, user_text: str) -> str:
        """One-shot, non-streaming, tool-free completion — used to generate a
        summary of old history during compaction. Must be overridden by any
        backend that wants to support compaction."""
        raise NotImplementedError(f"{self.name} backend does not support context summarization")
