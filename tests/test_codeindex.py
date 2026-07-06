import json
import os
import time
from pathlib import Path

import tythancode.codeindex as codeindex
from tythancode.codeindex import CACHE_VERSION, _load_cache, build_index, format_hits, search_index, tokenize
from tythancode.tools import Workspace


def test_tokenize_splits_snake_case():
    tokens = tokenize("def read_file(path):")
    assert "read_file" in tokens
    assert "read" in tokens
    assert "file" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("function readFile(path) {}")
    assert "readfile" in tokens
    assert "read" in tokens
    assert "file" in tokens


def test_tokenize_drops_single_char_tokens():
    assert "a" not in tokenize("a = b + c")


def test_build_index_finds_relevant_chunk(tmp_path):
    (tmp_path / "net.py").write_text(
        "\n".join(f"line {i}" for i in range(30))
        + "\ndef retry_with_backoff(fn):\n    \"\"\"Retries a network call with exponential backoff.\"\"\"\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text("def render_template(name):\n    return name\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)

    hits = search_index(index, "retry backoff network", top_k=5)
    assert hits
    assert hits[0].path == "net.py"


def test_search_no_match_returns_empty(tmp_path):
    (tmp_path / "a.py").write_text("print('hello world')\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)
    assert search_index(index, "quantum teleportation") == []


def test_format_hits_empty():
    assert "no matches" in format_hits([])


def test_format_hits_includes_path_and_lines(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)
    hits = search_index(index, "foo")
    out = format_hits(hits)
    assert "a.py:1-2" in out


def test_build_index_skips_binary_files(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "code.py").write_text("def handler(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)
    assert all(c.path != "data.bin" for c in index.chunks)


def test_build_index_respects_gitignore(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("def vendored_thing(): pass\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def app_thing(): pass\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("vendor/\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)
    assert all(c.path != "vendor/lib.py" for c in index.chunks)
    assert any(c.path == "app.py" for c in index.chunks)


def test_chunking_splits_long_files(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(200)), encoding="utf-8")
    ws = Workspace(tmp_path)
    index = build_index(ws)
    chunks = [c for c in index.chunks if c.path == "big.py"]
    assert len(chunks) > 1
    assert chunks[0].start_line == 1


# -- persistent cache ------------------------------------------------------


def test_no_cache_dir_never_touches_disk(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    build_index(ws)  # no cache_dir
    assert list(tmp_path.glob("**/index.json")) == []


def test_cache_is_written_to_disk(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    cache_dir = tmp_path / ".cache"
    build_index(ws, cache_dir=cache_dir)
    assert (cache_dir / "index.json").is_file()


def test_cache_reuses_chunks_for_unchanged_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    cache_dir = tmp_path / ".cache"

    index1 = build_index(ws, cache_dir=cache_dir)
    assert any(c.path == "a.py" for c in index1.chunks)

    calls = []
    original = codeindex._chunk_file

    def spy(rel_path, text):
        calls.append(rel_path)
        return original(rel_path, text)

    monkeypatch.setattr(codeindex, "_chunk_file", spy)
    index2 = build_index(ws, cache_dir=cache_dir)

    assert calls == []  # a.py's chunks came from the cache, not re-tokenized
    assert [c.text for c in index2.chunks] == [c.text for c in index1.chunks]


def test_cache_detects_changed_file_content(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("def foo(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    cache_dir = tmp_path / ".cache"
    build_index(ws, cache_dir=cache_dir)

    path.write_text("def a_totally_different_and_longer_name(): pass\n", encoding="utf-8")
    # Force the mtime to differ regardless of filesystem timestamp resolution.
    bumped = time.time() + 5
    os.utime(path, (bumped, bumped))

    index2 = build_index(ws, cache_dir=cache_dir)
    texts = [c.text for c in index2.chunks if c.path == "a.py"]
    assert any("a_totally_different_and_longer_name" in t for t in texts)
    assert not any(t == "def foo(): pass" for t in texts)


def test_cache_drops_files_that_no_longer_exist(tmp_path):
    keep = tmp_path / "keep.py"
    gone = tmp_path / "gone.py"
    keep.write_text("def keep_fn(): pass\n", encoding="utf-8")
    gone.write_text("def gone_fn(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    cache_dir = tmp_path / ".cache"
    build_index(ws, cache_dir=cache_dir)

    gone.unlink()
    index2 = build_index(ws, cache_dir=cache_dir)
    paths = {c.path for c in index2.chunks}
    assert "keep.py" in paths
    assert "gone.py" not in paths


def test_corrupted_cache_file_falls_back_to_full_rebuild(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    (cache_dir / "index.json").write_text("{not valid json", encoding="utf-8")

    index = build_index(ws, cache_dir=cache_dir)
    assert any(c.path == "a.py" for c in index.chunks)


def test_load_cache_rejects_wrong_version(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": CACHE_VERSION - 1, "files": {}}), encoding="utf-8")
    assert _load_cache(path) == {}


def test_load_cache_rejects_missing_file():
    assert _load_cache(Path("/nonexistent/path/index.json")) == {}
