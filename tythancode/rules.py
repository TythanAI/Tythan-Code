"""Project-level instructions, auto-loaded into the system prompt.

Mirrors the "rules file" convention other AI coding tools already use
(Cursor's `.cursorrules`, the cross-tool `AGENTS.md` standard, ...): a
plain-text file the user maintains once at the workspace root — conventions,
build/test commands, things the agent should or shouldn't touch — instead of
repeating the same context in chat every session.

Only the first matching file is used; merging several would risk duplicate
or conflicting instructions with no clear precedence. Priority goes from
most Tythan-Code-specific to the more generic, cross-tool conventions, so a
project that deliberately writes Tythan-Code-specific guidance gets it over
a generic AGENTS.md meant for other tools too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANDIDATE_FILENAMES = (
    ".tythancode/rules.md",
    ".tythancoderules",
    "AGENTS.md",
    ".cursorrules",
)

# Generous but bounded: this is prepended to every system prompt, so an
# unbounded file would eat into the actual conversation's context budget on
# every single call.
MAX_RULES_CHARS = 10_000


@dataclass
class ProjectRules:
    source: str  # filename, relative to the workspace root
    text: str


def load_project_rules(root: Path) -> ProjectRules | None:
    """Return the first rules file found under `root`, or None if none exist
    or every candidate is empty/unreadable."""
    for name in CANDIDATE_FILENAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > MAX_RULES_CHARS:
            text = text[:MAX_RULES_CHARS] + "\n... [truncated]"
        return ProjectRules(source=name, text=text)
    return None
