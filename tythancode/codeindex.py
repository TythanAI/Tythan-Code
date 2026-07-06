"""Lightweight local codebase index for ranked ("fuzzy") search.

Plain `search` (in tools.py) is exact regex over lines — great when the model
already knows the string or pattern to look for, useless when it doesn't
("where does this project handle retry logic for network calls?"). This
module closes that gap with BM25 (the same ranking family classic search
engines use) over fixed-size chunks of every indexed file, honestly labeled:
it is lexical/keyword ranking, not embeddings-based semantic search — no
model call, no vector database, no network, works fully offline. The
tokenizer also splits camelCase and snake_case identifiers into parts (so a
query for "read file" scores `read_file` and `readFile` too), which recovers
a good chunk of what a naive keyword search would otherwise miss in code.

Rebuilding from scratch is fine for a one-off search, but paying it again on
every session start on a large project adds up. So `build_index` can also
take a `cache_dir`: a JSON file recording each file's chunks alongside its
`(mtime, size)`, so a file whose stat hasn't changed since the last build
reuses its cached chunks instead of being re-read and re-tokenized. This is
an optimization only — the cache is versioned (`CACHE_VERSION`), so a future
change to chunking or tokenization invalidates it automatically rather than
silently serving chunks built under different rules, and any read/parse/
write failure just falls back to a full rebuild rather than propagating.
Passing no `cache_dir` (the default) skips the cache entirely: build fresh,
touch no disk. `Agent` drops its in-memory instance after any tool call that
could have changed files, so the next search always reflects current
content — the persistent cache only saves re-reading files that didn't
actually change.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .security import MAX_FILE_BYTES, SKIP_FILENAMES, SKIP_NAME_PARTS

# Bump when chunking or tokenization logic changes, so old on-disk caches
# (built under different rules) are ignored instead of silently misused.
CACHE_VERSION = 1

# Cap how much of a large project gets indexed so a single search call can't
# hang on a huge monorepo. A search against a partial index still returns
# the best matches found within it, just with a note that it's partial.
MAX_INDEXED_FILES = 5_000
MAX_INDEXED_CHUNKS = 40_000
CHUNK_LINES = 50

# Okapi BM25 parameters — standard defaults, not tuned per-project.
K1 = 1.5
B = 0.75

_WORD_RX = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RX = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. snake_case and camelCase identifiers are also
    split into their parts, in addition to the whole identifier, so a query
    for "read file" matches `read_file`/`readFile` chunks too."""
    tokens: list[str] = []
    for word in _WORD_RX.findall(text):
        tokens.append(word.lower())
        if "_" in word:
            tokens.extend(p.lower() for p in word.split("_") if p)
        camel_parts = _CAMEL_BOUNDARY_RX.split(word)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts if p)
    return [t for t in tokens if len(t) > 1]


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    term_freq: dict[str, int] = field(default_factory=dict)
    length: int = 0  # token count, for BM25's length normalization


@dataclass
class CodeIndex:
    chunks: list[Chunk]
    doc_freq: dict[str, int]  # token -> number of chunks containing it
    avg_length: float
    files_indexed: int
    files_skipped_limit: bool = False


def _chunk_file(rel_path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), CHUNK_LINES):
        window = lines[start : start + CHUNK_LINES]
        if not window:
            continue
        body = "\n".join(window)
        tokens = tokenize(body)
        if not tokens:
            continue
        term_freq: dict[str, int] = {}
        for t in tokens:
            term_freq[t] = term_freq.get(t, 0) + 1
        chunks.append(Chunk(
            path=rel_path,
            start_line=start + 1,
            end_line=start + len(window),
            text=body,
            term_freq=term_freq,
            length=len(tokens),
        ))
    return chunks


@dataclass
class _CachedFile:
    mtime: float
    size: int
    chunks: list[Chunk]


def _cache_file_path(cache_dir: Path) -> Path:
    return cache_dir / "index.json"


def _load_cache(path: Path) -> dict[str, _CachedFile]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    out: dict[str, _CachedFile] = {}
    try:
        for rel, entry in raw.get("files", {}).items():
            chunks = [
                Chunk(
                    path=rel,
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    text=c["text"],
                    term_freq=c["term_freq"],
                    length=c["length"],
                )
                for c in entry["chunks"]
            ]
            out[rel] = _CachedFile(mtime=entry["mtime"], size=entry["size"], chunks=chunks)
    except (KeyError, TypeError, ValueError):
        return {}  # malformed cache — rebuild fully rather than half-trust it
    return out


def _save_cache(path: Path, files_map: dict[str, _CachedFile]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION,
            "files": {
                rel: {
                    "mtime": entry.mtime,
                    "size": entry.size,
                    "chunks": [
                        {
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "text": c.text,
                            "term_freq": c.term_freq,
                            "length": c.length,
                        }
                        for c in entry.chunks
                    ],
                }
                for rel, entry in files_map.items()
            },
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # caching is a pure optimization; never let it break indexing


def _aggregate(chunks: list[Chunk]) -> tuple[dict[str, int], float]:
    doc_freq: dict[str, int] = {}
    for c in chunks:
        for t in c.term_freq:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    avg_length = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0
    return doc_freq, avg_length


def build_index(ws, cache_dir: Path | None = None) -> CodeIndex:
    """Build an index over every non-ignored, non-binary file in workspace
    `ws` (a `tools.Workspace`). With `cache_dir` set, a file whose (mtime,
    size) matches the last build reuses its cached chunks instead of being
    re-read and re-tokenized; without it, always builds fresh in memory."""
    cache_path = _cache_file_path(cache_dir) if cache_dir is not None else None
    cached_files = _load_cache(cache_path) if cache_path is not None else {}
    # Defensive: if the cache directory happens to live inside the workspace
    # being indexed (it doesn't in production — Tythan Code's default is
    # under ~/.tythancode, outside any project — but callers can pass
    # anything), never index the cache's own file. Otherwise every rebuild
    # would "discover" its own previous output as source content.
    cache_dir_resolved = cache_dir.resolve() if cache_dir is not None else None

    files_map: dict[str, _CachedFile] = {}
    files_indexed = 0
    total_chunks = 0
    files_skipped_limit = False

    for p in sorted(ws.root.rglob("*")):
        if not p.is_file():
            continue
        if cache_dir_resolved is not None and p.resolve().is_relative_to(cache_dir_resolved):
            continue
        rel = p.relative_to(ws.root)
        if ws.is_ignored(rel):
            continue
        if p.name in SKIP_FILENAMES or any(part in p.name for part in SKIP_NAME_PARTS):
            continue
        if files_indexed >= MAX_INDEXED_FILES or total_chunks >= MAX_INDEXED_CHUNKS:
            files_skipped_limit = True
            break
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size > MAX_FILE_BYTES:
            continue

        rel_str = str(rel)
        cached = cached_files.get(rel_str)
        if cached is not None and cached.mtime == st.st_mtime and cached.size == st.st_size:
            file_chunks = cached.chunks
        else:
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:1024]:
                continue  # binary
            text = raw.decode("utf-8", errors="replace")
            file_chunks = _chunk_file(rel_str, text)

        if file_chunks:
            files_map[rel_str] = _CachedFile(mtime=st.st_mtime, size=st.st_size, chunks=file_chunks)
            files_indexed += 1
            total_chunks += len(file_chunks)

    chunks = [c for entry in files_map.values() for c in entry.chunks]
    doc_freq, avg_length = _aggregate(chunks)
    if cache_path is not None:
        _save_cache(cache_path, files_map)
    return CodeIndex(
        chunks=chunks,
        doc_freq=doc_freq,
        avg_length=avg_length,
        files_indexed=files_indexed,
        files_skipped_limit=files_skipped_limit,
    )


def _bm25_score(chunk: Chunk, query_terms: set[str], index: CodeIndex) -> float:
    n = len(index.chunks)
    score = 0.0
    for term in query_terms:
        freq = chunk.term_freq.get(term, 0)
        if freq == 0:
            continue
        df = index.doc_freq.get(term, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        norm = 1 - B + B * (chunk.length / index.avg_length if index.avg_length else 1)
        score += idf * (freq * (K1 + 1)) / (freq + K1 * norm)
    return score


@dataclass
class SearchHit:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str


def search_index(index: CodeIndex, query: str, top_k: int = 8) -> list[SearchHit]:
    query_terms = set(tokenize(query))
    if not query_terms or not index.chunks:
        return []
    scored = []
    for chunk in index.chunks:
        score = _bm25_score(chunk, query_terms, index)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        SearchHit(path=c.path, start_line=c.start_line, end_line=c.end_line,
                  score=round(score, 3), snippet=c.text)
        for score, c in scored[: max(top_k, 0)]
    ]


def format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no matches — try different or broader terms, or use `search` for an exact regex)"
    parts = []
    for h in hits:
        parts.append(f"{h.path}:{h.start_line}-{h.end_line} (relevance {h.score})\n{h.snippet}")
    return "\n\n".join(parts)
