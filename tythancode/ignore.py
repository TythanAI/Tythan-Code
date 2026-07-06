"""Minimal `.gitignore` matching, so tools see the same "ignored" files a
real git repo would hide.

`SKIP_DIRS` in tools.py only knows about a handful of hardcoded junk
directories (.git, node_modules, ...). Real projects routinely have their own
build/output/coverage directories that aren't in that list, which means they
leak into `list_files`, `search` and the security scanner as noise (or worse,
generate bogus security findings on vendored/generated code). Respecting the
project's own `.gitignore` fixes that without having to guess every possible
build tool's output directory name.

This implements the common subset of gitignore syntax — comments, blank
lines, `!` negation, trailing `/` for directory-only patterns, a leading `/`
to anchor to the gitignore's location, and `*`/`**`/`?` globs — which covers
the overwhelming majority of real-world `.gitignore` files. It does not
implement the full spec (e.g. character classes like `[a-z]` are treated
literally). Only the workspace root's `.gitignore` is read; nested
per-directory `.gitignore` files are not merged in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Pattern:
    regex: re.Pattern
    negate: bool
    dir_only: bool


def _translate(pattern: str) -> re.Pattern:
    """Translate one gitignore glob pattern into a compiled regex.

    Matches against a path's parts already joined with '/' (no leading or
    trailing slash). `**` matches across directory boundaries; a lone `*`
    does not.
    """
    i, n = 0, len(pattern)
    out = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                # "**/" -> match zero or more path segments.
                if pattern[i : i + 3] == "**/":
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if c == "?":
            out.append("[^/]")
            i += 1
            continue
        out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _compile_pattern(raw: str) -> _Pattern | None:
    line = raw.rstrip("\n")
    if not line or line.lstrip().startswith("#"):
        return None
    line = line.rstrip()
    if not line:
        return None
    # Unescape a leading "\#" or "\!" (literal, not comment/negation).
    negate = False
    if line.startswith("!"):
        negate = True
        line = line[1:]
    if line.startswith("\\"):
        line = line[1:]
    dir_only = line.endswith("/")
    if dir_only:
        line = line[:-1]
    anchored = line.startswith("/")
    if anchored:
        line = line[1:]
    if not line:
        return None
    body = _translate(line).pattern[1:-1]  # strip the ^...$ we'll re-add below
    if anchored or "/" in line:
        regex = re.compile(f"^{body}$")
    else:
        # Unanchored pattern with no inner slash: matches at any depth.
        regex = re.compile(f"(^|.*/){body}$")
    return _Pattern(regex=regex, negate=negate, dir_only=dir_only)


@dataclass
class GitignoreMatcher:
    """Matches workspace-relative POSIX paths against a set of gitignore rules."""

    patterns: list[_Pattern] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "GitignoreMatcher":
        patterns: list[_Pattern] = []
        for name in (".gitignore", ".git/info/exclude"):
            p = root / name
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                compiled = _compile_pattern(line)
                if compiled is not None:
                    patterns.append(compiled)
        return cls(patterns=patterns)

    def matches(self, rel_path: str, is_dir: bool = False) -> bool:
        """`rel_path` is workspace-relative, POSIX-separated, no leading slash.

        Mirrors git's actual behavior: if an ancestor directory is ignored,
        everything under it is ignored too (git never descends into an
        ignored directory to evaluate negations for its contents). Only once
        no ancestor is ignored do we check the path itself.
        """
        if not self.patterns:
            return False
        parts = rel_path.split("/")
        for i in range(len(parts) - 1):
            if self._matches_one("/".join(parts[: i + 1]), is_dir=True):
                return True
        return self._matches_one(rel_path, is_dir=is_dir)

    def _matches_one(self, path: str, is_dir: bool) -> bool:
        """Last-match-wins over patterns applicable to a single path segment."""
        ignored = False
        for pat in self.patterns:
            if pat.dir_only and not is_dir:
                continue
            if pat.regex.match(path):
                ignored = not pat.negate
        return ignored
