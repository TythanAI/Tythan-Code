import pytest

from tythancode.tools import ToolError, Workspace, validate_todos


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\nhello world\n", encoding="utf-8")
    return Workspace(tmp_path)


def test_read_file(ws):
    out = ws.read_file("src/app.py")
    assert "1\tdef main():" in out
    assert "print('hello')" in out


def test_read_missing_file(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.read_file("nope.py")


def test_path_traversal_blocked(ws):
    with pytest.raises(ToolError, match="escapes"):
        ws.read_file("../../etc/passwd")
    with pytest.raises(ToolError, match="escapes"):
        ws.resolve("/etc/passwd")


def test_write_and_overwrite(ws, tmp_path):
    msg = ws.write_file("new/dir/file.txt", "content")
    assert "file.txt" in msg
    assert (tmp_path / "new" / "dir" / "file.txt").read_text() == "content"
    ws.write_file("new/dir/file.txt", "changed")
    assert (tmp_path / "new" / "dir" / "file.txt").read_text() == "changed"


def test_edit_file(ws, tmp_path):
    ws.edit_file("src/app.py", "print('hello')", "print('bye')")
    assert "print('bye')" in (tmp_path / "src" / "app.py").read_text()


def test_edit_requires_unique_match(ws, tmp_path):
    (tmp_path / "dup.txt").write_text("aaa\naaa\n")
    with pytest.raises(ToolError, match="2 times"):
        ws.edit_file("dup.txt", "aaa", "bbb")
    ws.edit_file("dup.txt", "aaa", "bbb", replace_all=True)
    assert (tmp_path / "dup.txt").read_text() == "bbb\nbbb\n"


def test_edit_missing_string(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.edit_file("src/app.py", "no such text", "x")


def test_list_files(ws):
    out = ws.list_files("**/*.py")
    assert "src/app.py" in out
    assert "README.md" not in out


def test_list_skips_junk_dirs(ws, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    assert ".git" not in ws.list_files("**/*")


def test_search(ws):
    out = ws.search(r"hello", glob="**/*.py")
    assert "src/app.py:2" in out
    assert "README.md" not in out


def test_search_invalid_regex(ws):
    with pytest.raises(ToolError, match="Invalid regex"):
        ws.search("([")


def test_run_command(ws):
    out = ws.run_command("echo hi")
    assert "hi" in out
    assert "[exit code: 0]" in out


def test_run_command_nonzero_exit(ws):
    out = ws.run_command("false")
    assert "[exit code: 1]" in out


def test_validate_todos_normalizes_valid_input():
    todos = validate_todos([
        {"content": "  step one  ", "status": "completed"},
        {"content": "step two", "status": "in_progress"},
    ])
    assert todos == [
        {"content": "step one", "status": "completed"},
        {"content": "step two", "status": "in_progress"},
    ]


def test_validate_todos_rejects_non_list():
    with pytest.raises(ToolError, match="must be a list"):
        validate_todos({"content": "x", "status": "pending"})


def test_validate_todos_rejects_bad_status():
    with pytest.raises(ToolError, match="invalid status"):
        validate_todos([{"content": "x", "status": "done"}])


def test_validate_todos_rejects_empty_content():
    with pytest.raises(ToolError, match="non-empty"):
        validate_todos([{"content": "  ", "status": "pending"}])


def test_validate_todos_rejects_multiple_in_progress():
    with pytest.raises(ToolError, match="only one todo"):
        validate_todos([
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ])
