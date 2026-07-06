from tythancode.ignore import GitignoreMatcher
from tythancode.tools import Workspace


def write_gitignore(tmp_path, text):
    (tmp_path / ".gitignore").write_text(text, encoding="utf-8")


def test_no_gitignore_matches_nothing(tmp_path):
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("anything.py") is False


def test_simple_filename_pattern(tmp_path):
    write_gitignore(tmp_path, "*.log\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("debug.log")
    assert m.matches("nested/dir/debug.log")
    assert not m.matches("debug.txt")


def test_directory_pattern_hides_whole_subtree(tmp_path):
    write_gitignore(tmp_path, "build/\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("build", is_dir=True)
    assert m.matches("build/output.js")
    assert m.matches("build/nested/deep/file.txt")
    assert not m.matches("rebuild/output.js")


def test_anchored_pattern_only_matches_root(tmp_path):
    write_gitignore(tmp_path, "/dist\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("dist", is_dir=True)
    assert not m.matches("src/dist", is_dir=True)


def test_negation_reincludes_specific_file(tmp_path):
    write_gitignore(tmp_path, "*.log\n!keep.log\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("debug.log")
    assert not m.matches("keep.log")


def test_negation_inside_ignored_directory_has_no_effect(tmp_path):
    # Matches real git behavior: git never descends into an ignored directory
    # to evaluate negations for files inside it.
    write_gitignore(tmp_path, "build/\n!build/keep.txt\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("build/keep.txt")


def test_double_star_matches_any_depth(tmp_path):
    write_gitignore(tmp_path, "**/__generated__/**\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("a/__generated__/b/c.py")


def test_comments_and_blank_lines_ignored(tmp_path):
    write_gitignore(tmp_path, "# comment\n\n*.log\n")
    m = GitignoreMatcher.load(tmp_path)
    assert m.matches("x.log")


def test_workspace_list_files_respects_gitignore(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print(1)\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "bundle.js").write_text("noise\n")
    write_gitignore(tmp_path, "build/\n")
    ws = Workspace(tmp_path)
    out = ws.list_files("**/*")
    assert "src/app.py" in out
    assert "build" not in out


def test_workspace_search_respects_gitignore(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("password = 'hunter2'\n")
    (tmp_path / "app.py").write_text("password = 'hunter2'\n")
    write_gitignore(tmp_path, "vendor/\n")
    ws = Workspace(tmp_path)
    out = ws.search("password")
    assert "app.py" in out
    assert "vendor" not in out
