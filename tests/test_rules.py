from tythancode.rules import MAX_RULES_CHARS, load_project_rules


def test_no_rules_file_returns_none(tmp_path):
    assert load_project_rules(tmp_path) is None


def test_loads_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use pytest for tests.\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules is not None
    assert rules.source == "AGENTS.md"
    assert "pytest" in rules.text


def test_tythancoderules_takes_priority_over_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("generic instructions\n", encoding="utf-8")
    (tmp_path / ".tythancoderules").write_text("tythan-code specific\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules.source == ".tythancoderules"
    assert "tythan-code specific" in rules.text


def test_nested_rules_md_takes_top_priority(tmp_path):
    (tmp_path / ".tythancode").mkdir()
    (tmp_path / ".tythancode" / "rules.md").write_text("top priority\n", encoding="utf-8")
    (tmp_path / ".tythancoderules").write_text("lower priority\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules.source == ".tythancode/rules.md"


def test_cursorrules_fallback(tmp_path):
    (tmp_path / ".cursorrules").write_text("legacy cursor rules\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules.source == ".cursorrules"


def test_empty_file_is_skipped_in_favor_of_next_candidate(tmp_path):
    (tmp_path / ".tythancoderules").write_text("   \n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("real content\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules.source == "AGENTS.md"


def test_oversized_file_is_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * (MAX_RULES_CHARS + 500), encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert len(rules.text) <= MAX_RULES_CHARS + len("\n... [truncated]")
    assert rules.text.endswith("[truncated]")


def test_directory_named_like_a_candidate_is_not_treated_as_a_file(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()
    (tmp_path / ".cursorrules").write_text("fallback\n", encoding="utf-8")
    rules = load_project_rules(tmp_path)
    assert rules.source == ".cursorrules"
