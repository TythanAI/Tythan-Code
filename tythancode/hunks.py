"""Per-hunk diff review: split a proposed file change into independent
pieces so the user can accept some and reject others, instead of an
all-or-nothing yes/no on the whole file.

Built directly on `difflib.SequenceMatcher`'s opcodes rather than parsing a
unified diff back apart: each non-'equal' opcode is already an independent,
exactly-reversible edit (replace/insert/delete over a precise line range),
so reconstructing the final content from a subset of accepted opcodes is
just picking the old-side or new-side lines per opcode — no context-line
bookkeeping, no ambiguity about where a hunk begins or ends, and the result
is guaranteed to be exactly `old_text` or exactly `new_text` at every
position depending on what was accepted.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class Hunk:
    tag: str  # 'replace' | 'insert' | 'delete' (difflib's opcode tags, minus 'equal')
    old_lines: list[str]
    new_lines: list[str]
    old_start: int  # 1-based line number in the old file
    new_start: int  # 1-based line number in the new file


def _opcodes(old_text: str, new_text: str):
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    return old_lines, new_lines, matcher.get_opcodes()


def split_hunks(old_text: str, new_text: str) -> list[Hunk]:
    """Return the independent, non-'equal' change regions between old and new."""
    old_lines, new_lines, opcodes = _opcodes(old_text, new_text)
    hunks = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        hunks.append(Hunk(
            tag=tag,
            old_lines=old_lines[i1:i2],
            new_lines=new_lines[j1:j2],
            old_start=i1 + 1,
            new_start=j1 + 1,
        ))
    return hunks


def apply_selected_hunks(old_text: str, new_text: str, accepted: list[bool]) -> str:
    """Reconstruct file content applying only the hunks whose corresponding
    `accepted` flag is True, in the same order `split_hunks` returns them;
    a rejected hunk keeps the old side. `accepted` must have exactly one
    entry per hunk `split_hunks(old_text, new_text)` would return."""
    old_lines, new_lines, opcodes = _opcodes(old_text, new_text)
    result: list[str] = []
    hunk_index = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            result.extend(old_lines[i1:i2])
            continue
        if hunk_index >= len(accepted):
            raise ValueError("accepted has fewer entries than there are hunks")
        result.extend(new_lines[j1:j2] if accepted[hunk_index] else old_lines[i1:i2])
        hunk_index += 1
    if hunk_index != len(accepted):
        raise ValueError("accepted has more entries than there are hunks")
    return "".join(result)
