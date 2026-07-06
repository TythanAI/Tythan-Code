"""Tool definitions and executors.

Every file operation is confined to the workspace root: model-supplied paths
are resolved and rejected if they escape it.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

from .config import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_READ_LINES,
    MAX_TOOL_OUTPUT_CHARS,
)
from .ignore import GitignoreMatcher

# Directories we never descend into when listing/searching.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".ruff_cache", "dist", "build", ".pytest_cache"}

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the workspace. Returns the content with line numbers. "
            "Call this before editing a file. Use offset/limit for large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "offset": {"type": "integer", "description": "1-based line to start reading from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file in the workspace with the given content. "
            "The user sees a diff and confirms before the write happens. "
            "Always output the complete file content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "content": {"type": "string", "description": "Full new file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. old_string must appear exactly once "
            "(or set replace_all to true). Read the file first so old_string matches exactly, "
            "including whitespace. The user sees a diff and confirms before the edit happens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in the workspace matching a glob pattern (e.g. '**/*.py', 'src/*.ts'). "
            "Defaults to listing everything. Common junk directories (.git, node_modules, ...) are skipped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, default '**/*'"},
            },
        },
    },
    {
        "name": "search",
        "description": (
            "Search file contents in the workspace with a regular expression. "
            "Returns matching lines as path:line:text. Use glob to narrow which files are searched."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '**/*.py'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "code_search",
        "description": (
            "Rank-search the whole codebase by relevance to a query, for when you don't "
            "know the exact string or regex to look for — e.g. 'where does retry logic for "
            "network calls live' or 'user authentication flow'. Uses keyword ranking "
            "(BM25) over chunked files, not embeddings — it rewards files that use similar "
            "words, so it works best with a few descriptive keywords rather than a full "
            "sentence. Returns the best-matching chunks with path and line range. Prefer "
            "`search` instead when you already know the exact text or a precise regex."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or a short natural-language description"},
                "top_k": {"type": "integer", "description": "Number of results to return (default 8, max 20)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch a web page or API response over HTTP(S) and return its readable text "
            "(HTML is stripped down to plain text). Use this to read documentation, check "
            "a changelog, or pull in reference material you don't already know. Only "
            "public internet addresses are reachable — internal/private/loopback "
            "addresses are refused. The user confirms before the request is made, since "
            "it reaches outside the workspace onto the network."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http:// or https:// URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "security_scan",
        "description": (
            "Scan the workspace (or a subdirectory/file) for security issues: leaked "
            "secrets and API keys, dangerous code patterns (eval, pickle, SQL built from "
            "f-strings, shell=True, disabled TLS verification, ...) and insecure config "
            "(wildcard CORS, JWT 'none', debug mode). Run this after writing or changing "
            "code, and when the user asks for a security review. Returns findings with "
            "severity, file and line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Subdirectory or file to scan (default: whole workspace)"},
                "include_dependencies": {
                    "type": "boolean",
                    "description": "Also check pinned dependencies against the OSV.dev vulnerability database (needs network, default false)",
                },
            },
        },
    },
    {
        "name": "todo_write",
        "description": (
            "Replace the current task plan with this updated list of steps. Use it for "
            "multi-step work (roughly 3+ distinct steps): write the full plan before "
            "starting, keep exactly one item 'in_progress' at a time, and mark an item "
            "'completed' as soon as it's actually done — not in a batch at the end. "
            "Skip this entirely for small or single-step requests. This is a pure "
            "progress-tracking tool: it never touches files and needs no confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The full plan, replacing whatever was there before",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Short description of the step"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the workspace root and return its output "
            "(stdout + stderr + exit code). The user confirms before the command runs. "
            "Use for tests, builds, git, installs, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
            },
            "required": ["command"],
        },
    },
]

VALID_TODO_STATUSES = {"pending", "in_progress", "completed"}


class ToolError(Exception):
    """Raised by tool executors; reported back to the model as is_error."""


def validate_todos(raw: object) -> list[dict]:
    """Validate and normalize todo_write's input. Raises ToolError (reported
    back to the model, so it can retry) on anything malformed."""
    if not isinstance(raw, list):
        raise ToolError("todos must be a list")
    todos = []
    in_progress_count = 0
    for item in raw:
        if not isinstance(item, dict):
            raise ToolError("each todo must be an object with 'content' and 'status'")
        content = item.get("content")
        status = item.get("status")
        if not isinstance(content, str) or not content.strip():
            raise ToolError("each todo needs a non-empty 'content' string")
        if status not in VALID_TODO_STATUSES:
            raise ToolError(f"invalid status {status!r} — must be one of {sorted(VALID_TODO_STATUSES)}")
        if status == "in_progress":
            in_progress_count += 1
        todos.append({"content": content.strip(), "status": status})
    if in_progress_count > 1:
        raise ToolError("only one todo can be 'in_progress' at a time")
    return todos


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


class Workspace:
    """Executes tools against a directory, confining all paths inside it."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        # Snapshotted once per session: a .gitignore edited mid-session won't
        # be picked up until Tythan Code restarts, same tradeoff the rest of
        # the codebase makes for provider/context-window config.
        self.ignore = GitignoreMatcher.load(self.root)

    def is_ignored(self, rel: Path) -> bool:
        """True if `rel` (relative to the workspace root) should be treated as
        absent by file-discovery tools: skipped junk dirs, or gitignored."""
        if any(part in SKIP_DIRS for part in rel.parts):
            return True
        return self.ignore.matches(rel.as_posix(), is_dir=(self.root / rel).is_dir())

    def resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"Path escapes the workspace: {path}")
        return candidate

    # -- read-only tools ---------------------------------------------------

    def read_file(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"File not found: {path}")
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise ToolError(f"Cannot read {path}: {exc}") from exc
        if offset < 1:
            offset = 1
        window = lines[offset - 1 : offset - 1 + limit]
        if not window and lines:
            raise ToolError(f"offset {offset} is beyond end of file ({len(lines)} lines)")
        numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(window, start=offset))
        suffix = ""
        remaining = len(lines) - (offset - 1 + len(window))
        if remaining > 0:
            suffix = f"\n... [{remaining} more lines — use offset={offset + len(window)} to continue]"
        return truncate(numbered + suffix) if numbered else "(empty file)"

    def list_files(self, pattern: str = "**/*") -> str:
        results = []
        for p in sorted(self.root.glob(pattern)):
            rel = p.relative_to(self.root)
            if self.is_ignored(rel):
                continue
            if p.is_file():
                results.append(str(rel))
            if len(results) >= 500:
                results.append("... [more files omitted, narrow the pattern]")
                break
        return "\n".join(results) if results else "(no files match)"

    def search(self, pattern: str, glob: str = "**/*") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        hits: list[str] = []
        for p in sorted(self.root.glob(glob)):
            rel = p.relative_to(self.root)
            if self.is_ignored(rel) or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text[:1024]:  # skip binaries
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{rel}:{lineno}:{line.strip()[:300]}")
                    if len(hits) >= 200:
                        hits.append("... [more matches omitted, narrow the search]")
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "(no matches)"

    # -- mutating tools ----------------------------------------------------

    def prepare_write(self, path: str, content: str) -> tuple[Path, str]:
        """Validate a write and return (target, old_content) for diffing."""
        target = self.resolve(path)
        old = ""
        if target.exists():
            if not target.is_file():
                raise ToolError(f"Not a regular file: {path}")
            old = target.read_text(encoding="utf-8", errors="replace")
        return target, old

    def write_file(self, path: str, content: str) -> str:
        self.prepare_write(path, content)
        return self.apply_content(path, content)

    def apply_content(self, path: str, content: str) -> str:
        """Write final content to `path` (already validated by the caller via
        prepare_write/prepare_edit). Used both by write_file directly and by
        the agent after per-hunk review, where the bytes actually written may
        be a merge of accepted/rejected hunks rather than exactly what
        write_file/edit_file were originally asked to write."""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def prepare_edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> tuple[Path, str, str]:
        """Validate an edit and return (target, old_content, new_content)."""
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"File not found: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            raise ToolError("old_string not found in file — read the file and match it exactly")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string appears {count} times — make it unique or set replace_all=true"
            )
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        return target, text, new_text

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        target, _, new_text = self.prepare_edit(path, old_string, new_string, replace_all)
        target.write_text(new_text, encoding="utf-8")
        return f"Edited {path}"

    def run_command(self, command: str) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s") from None
        parts = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr}")
        parts.append(f"[exit code: {proc.returncode}]")
        return truncate("\n".join(parts))


def matches_glob(rel_path: str, glob: str) -> bool:
    """Small helper used in tests; fnmatch-style match on a relative path."""
    return fnmatch.fnmatch(rel_path, glob)


MENTION_RX = re.compile(r"@([A-Za-z0-9_\-./]+)")


def expand_mentions(text: str, ws: Workspace) -> str:
    """Expand @path mentions: append the referenced files' contents to the message.

    Unresolvable mentions (emails, handles, missing files) are left untouched.
    """
    seen: list[str] = []
    for name in MENTION_RX.findall(text):
        if name in seen:
            continue
        seen.append(name)
        try:
            target = ws.resolve(name)
        except ToolError:
            continue
        if not target.is_file():
            continue
        content = truncate(target.read_text(encoding="utf-8", errors="replace"))
        text += f'\n\n<file path="{name}">\n{content}\n</file>'
    return text
