"""Model provider backends."""

from __future__ import annotations

from ..config import Config, ProviderConfig
from .base import Backend, BackendConfigError, ToolCall, ToolResult, TurnResult


def make_backend(pcfg: ProviderConfig, config: Config) -> Backend:
    if pcfg.type == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(pcfg, config)
    if pcfg.type == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(pcfg)
    raise BackendConfigError(f"unknown provider type: {pcfg.type!r} (use 'anthropic' or 'openai')")


__all__ = [
    "Backend",
    "BackendConfigError",
    "ToolCall",
    "ToolResult",
    "TurnResult",
    "make_backend",
]
