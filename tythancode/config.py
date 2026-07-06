"""Runtime configuration for Tythan Code.

Provider registry lives in ~/.tythancode/config.json (created with a template
on first run). Each provider entry is either the native Anthropic API or any
OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, DeepSeek, Ollama, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .mcp_client import MCPServerConfig

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_EFFORT = "high"  # low | medium | high | xhigh | max (Anthropic only)
MAX_TOKENS = 64000

# Limits for tool output so a single result can't blow up the context.
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_READ_LINES = 2_000
COMMAND_TIMEOUT_SECONDS = 120

# Hosts that never need an API key and are assumed to run small local models.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")

# Context-window defaults, used when a provider entry doesn't set its own
# "context_window". These are deliberately conservative guesses, not exact
# figures for any specific model — they only drive when Tythan Code proactively
# summarizes old history, so erring small (extra compaction) is far cheaper
# than erring large (a hard context-length error mid-turn). Override per
# provider with a "context_window" key in ~/.tythancode/config.json.
DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200_000
DEFAULT_LOCAL_CONTEXT_WINDOW = 8_000
DEFAULT_GENERIC_CONTEXT_WINDOW = 32_000
KNOWN_HOST_CONTEXT_WINDOWS: dict[str, int] = {
    "api.openai.com": 128_000,
    "openrouter.ai": 128_000,
    "api.groq.com": 128_000,
    "api.deepseek.com": 64_000,
    "api.mistral.ai": 128_000,
    "api.x.ai": 128_000,
}

CONFIG_DIR = Path.home() / ".tythancode"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "default_provider": "anthropic",
    "providers": {
        "anthropic": {
            "type": "anthropic",
            "model": DEFAULT_MODEL,
        },
        "openai": {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o",
        },
        "openrouter": {
            "type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "anthropic/claude-sonnet-4.5",
        },
        "ollama": {
            "type": "openai",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5-coder:14b",
        },
    },
    # External tool servers (MCP — Model Context Protocol), e.g.:
    # "mcp_servers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
    "mcp_servers": {},
}


@dataclass
class ProviderConfig:
    name: str
    type: str  # "anthropic" | "openai" (OpenAI-compatible)
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    context_window: int | None = None  # tokens; None means "use the type/host default"


def default_context_window(pcfg: ProviderConfig) -> int:
    """Best-effort context window for a provider that didn't set one explicitly."""
    if pcfg.context_window is not None:
        return pcfg.context_window
    if pcfg.type == "anthropic":
        return DEFAULT_ANTHROPIC_CONTEXT_WINDOW
    base_url = pcfg.base_url or ""
    if any(host in base_url for host in LOCAL_HOSTS):
        return DEFAULT_LOCAL_CONTEXT_WINDOW
    for host, window in KNOWN_HOST_CONTEXT_WINDOWS.items():
        if host in base_url:
            return window
    return DEFAULT_GENERIC_CONTEXT_WINDOW


def load_provider_configs(
    path: Path | None = None,
) -> tuple[str, dict[str, ProviderConfig]]:
    """Load (default_provider, providers) from the config file, creating it if missing."""
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    providers: dict[str, ProviderConfig] = {}
    for name, entry in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            type=entry.get("type", "openai"),
            model=entry.get("model", ""),
            base_url=entry.get("base_url"),
            api_key_env=entry.get("api_key_env"),
            context_window=entry.get("context_window"),
        )
    if not providers:
        raise ValueError(f"no providers defined in {path}")
    default = raw.get("default_provider")
    if default not in providers:
        default = next(iter(providers))
    return default, providers


def load_mcp_server_configs(path: Path | None = None) -> tuple[list[MCPServerConfig], list[str]]:
    """Load MCP server entries from the config file. Returns (configs, errors) —
    a malformed entry is reported as an error string and skipped, rather than
    silently dropped or crashing startup over one typo."""
    path = path or CONFIG_PATH
    if not path.exists():
        return [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"cannot read {path}: {exc}"]

    configs: list[MCPServerConfig] = []
    errors: list[str] = []
    for name, entry in (raw.get("mcp_servers") or {}).items():
        if not isinstance(entry, dict) or not entry.get("command"):
            errors.append(f"mcp_servers.{name}: missing required 'command' — skipped")
            continue
        args = entry.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            errors.append(f"mcp_servers.{name}: 'args' must be a list of strings — skipped")
            continue
        configs.append(MCPServerConfig(
            name=name,
            command=entry["command"],
            args=args,
            env=entry.get("env"),
        ))
    return configs, errors


@dataclass
class Config:
    workspace: Path
    effort: str = DEFAULT_EFFORT
    max_tokens: int = MAX_TOKENS
    yolo: bool = False  # skip confirmations for writes/commands
    checkpoints_enabled: bool = True  # record pre-edit file state for /undo
    compact_keep_rounds: int = 2  # most-recent user turns kept verbatim when compacting
