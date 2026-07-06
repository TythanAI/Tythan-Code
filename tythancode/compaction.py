"""Context-window compaction.

Long agent sessions accumulate an ever-growing message history until a
provider call fails outright because input + reserved output would exceed the
model's context window. This module provides the provider-agnostic pieces
needed to detect that situation and safely shrink the history: splitting the
flat message list into "rounds" (one real user turn plus everything the
agent did in response) and estimating how big the history roughly is when a
backend hasn't reported real token usage yet.

The actual summarization call and the decision of *when* to compact live in
Agent (agent.py) — this module only holds the pure, easily-testable logic.
"""

from __future__ import annotations

import json

# Rough fallback used only when a backend hasn't reported real token usage for
# the current history (e.g. an OpenAI-compatible endpoint that doesn't return
# usage on streamed responses). This is intentionally conservative-ish and
# provider-agnostic — good enough to trigger compaction before a hard context
# error, not an accounting-grade token count.
CHARS_PER_TOKEN_ESTIMATE = 4


def is_round_boundary(message: dict) -> bool:
    """True for a plain user turn: `{"role": "user", "content": "<str>"}`.

    Both backends append exactly this shape for real user input (see
    `Backend.add_user_message`), and something structurally different for
    tool results (Anthropic: role "user" but list content; OpenAI: role
    "tool"). So this shape reliably marks the start of a new round in either
    provider's native format without the caller needing to know which
    provider produced the history.
    """
    return message.get("role") == "user" and isinstance(message.get("content"), str)


def split_into_rounds(messages: list[dict]) -> list[list[dict]]:
    """Group a flat message list into rounds, each starting at a user-turn boundary.

    Any messages before the first boundary (shouldn't normally happen, since
    conversations always start with a user message) are kept together as a
    leading round so no message is ever dropped.
    """
    rounds: list[list[dict]] = []
    for m in messages:
        if is_round_boundary(m) or not rounds:
            rounds.append([m])
        else:
            rounds[-1].append(m)
    return rounds


def estimate_tokens_heuristic(messages: list[dict], system: str = "") -> int:
    """Rough token estimate (~4 chars/token) for when real usage isn't known."""
    try:
        body = json.dumps(messages, default=str, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        body = str(messages)
    return (len(body) + len(system)) // CHARS_PER_TOKEN_ESTIMATE


def cap_head(text: str, limit: int) -> str:
    """Keep the *tail* of `text` if it's over `limit` chars.

    Used to bound how much of the (potentially huge) old-rounds transcript we
    feed to the summarization call. The tail is kept rather than the head
    because the most recent old content is the most likely to still be
    relevant to what the agent should remember.
    """
    if len(text) <= limit:
        return text
    return f"[earlier content omitted]\n...{text[-limit:]}"
