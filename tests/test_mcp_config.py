import json

from tythancode.config import load_mcp_server_configs


def test_no_config_file_returns_empty(tmp_path):
    configs, errors = load_mcp_server_configs(tmp_path / "config.json")
    assert configs == []
    assert errors == []


def test_no_mcp_servers_key_returns_empty(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"default_provider": "anthropic", "providers": {}}))
    configs, errors = load_mcp_server_configs(path)
    assert configs == []
    assert errors == []


def test_valid_server_entry(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "mcp_servers": {
            "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {"X": "1"}},
        }
    }))
    configs, errors = load_mcp_server_configs(path)
    assert errors == []
    assert len(configs) == 1
    assert configs[0].name == "fetch"
    assert configs[0].command == "uvx"
    assert configs[0].args == ["mcp-server-fetch"]
    assert configs[0].env == {"X": "1"}


def test_missing_command_is_reported_and_skipped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mcp_servers": {"broken": {"args": ["x"]}}}))
    configs, errors = load_mcp_server_configs(path)
    assert configs == []
    assert len(errors) == 1
    assert "broken" in errors[0]
    assert "command" in errors[0]


def test_bad_args_type_is_reported_and_skipped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mcp_servers": {"broken": {"command": "x", "args": "not-a-list"}}}))
    configs, errors = load_mcp_server_configs(path)
    assert configs == []
    assert len(errors) == 1
    assert "args" in errors[0]


def test_one_bad_entry_does_not_block_others(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "mcp_servers": {
            "broken": {"args": ["x"]},
            "good": {"command": "uvx", "args": ["mcp-server-fetch"]},
        }
    }))
    configs, errors = load_mcp_server_configs(path)
    assert len(configs) == 1
    assert configs[0].name == "good"
    assert len(errors) == 1


def test_malformed_json_is_reported(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")
    configs, errors = load_mcp_server_configs(path)
    assert configs == []
    assert len(errors) == 1
