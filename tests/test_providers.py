"""Provider layer tests: config loading, tool conversion, mentions. Offline."""

import json

import pytest

from tythancode.config import Config, load_provider_configs
from tythancode.providers import BackendConfigError, make_backend
from tythancode.providers.openai_backend import _resolve_api_key, to_openai_tools
from tythancode.tools import TOOL_DEFINITIONS, Workspace, expand_mentions


def test_config_template_created(tmp_path):
    path = tmp_path / "config.json"
    default, providers = load_provider_configs(path)
    assert path.exists()
    assert default == "anthropic"
    assert {"anthropic", "openai", "openrouter", "ollama"} <= set(providers)
    assert providers["ollama"].base_url == "http://localhost:11434/v1"


def test_config_bad_default_falls_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "default_provider": "nope",
        "providers": {"only": {"type": "openai", "base_url": "http://localhost:1/v1", "model": "m"}},
    }))
    default, providers = load_provider_configs(path)
    assert default == "only"


def test_make_backend_rejects_unknown_type(tmp_path):
    from tythancode.config import ProviderConfig
    pcfg = ProviderConfig(name="x", type="wat", model="m")
    with pytest.raises(BackendConfigError, match="unknown provider type"):
        make_backend(pcfg, Config(workspace=tmp_path))


def test_openai_tool_conversion():
    tools = to_openai_tools(TOOL_DEFINITIONS)
    assert len(tools) == len(TOOL_DEFINITIONS)
    read = next(t for t in tools if t["function"]["name"] == "read_file")
    assert read["type"] == "function"
    assert "path" in read["function"]["parameters"]["properties"]


def test_api_key_resolution(monkeypatch):
    from tythancode.config import ProviderConfig

    monkeypatch.setenv("TEST_KEY", "sk-123")
    pcfg = ProviderConfig(name="p", type="openai", model="m",
                          base_url="https://api.example.com/v1", api_key_env="TEST_KEY")
    assert _resolve_api_key(pcfg) == "sk-123"

    monkeypatch.delenv("TEST_KEY")
    with pytest.raises(BackendConfigError, match="needs an API key"):
        _resolve_api_key(pcfg)

    local = ProviderConfig(name="ollama", type="openai", model="m",
                           base_url="http://localhost:11434/v1")
    assert _resolve_api_key(local) == "local"


def test_expand_mentions(tmp_path):
    (tmp_path / "notes.md").write_text("secret plans\n")
    ws = Workspace(tmp_path)

    out = expand_mentions("summarize @notes.md please", ws)
    assert 'file path="notes.md"' in out
    assert "secret plans" in out

    # non-files and escapes are ignored
    assert expand_mentions("email me @user.name", ws) == "email me @user.name"
    assert expand_mentions("look at @../../etc/passwd", ws) == "look at @../../etc/passwd"
