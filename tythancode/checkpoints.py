"""File-level checkpoints for /undo.

Before `write_file`/`edit_file` mutate a path, the agent records that path's
pre-turn content here. When the turn ends, everything recorded during it is
persisted as one checkpoint (a JSON file under `~/.tythancode/checkpoints/`,
keyed by a hash of the workspace path so unrelated projects don't collide).
`/undo` pops the most recent checkpoint and restores every file it touched to
its pre-turn state (or deletes it, if the file didn't exist before the turn).

Scope, on purpose: this only covers `write_file`/`edit_file`, the two tools
Tythan Code fully controls and already diffs before applying. `run_command`
can do anything (install packages, mutate unrelated state, talk to a
network) — there is no honest way to snapshot and revert that generically, so
it isn't covered. Checkpoints are a safety net for agent-authored file edits,
not a full VM undo.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .storage import workspace_key

CHECKPOINTS_ROOT = Path.home() / ".tythancode" / "checkpoints"

# Skip checkpointing files above this size (still allowed to be written, just
# not covered by /undo) so a single huge file can't blow up disk usage.
MAX_CHECKPOINT_FILE_BYTES = 5_000_000

# Retention cap per workspace so checkpoints don't accumulate forever.
MAX_CHECKPOINTS_PER_WORKSPACE = 50


@dataclass
class FileChange:
    path: str  # absolute path, resolved inside the workspace it was recorded for
    existed_before: bool
    before_content: str | None  # None means the file did not exist before this turn

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "existed_before": self.existed_before,
            "before_content": self.before_content,
        }

    @staticmethod
    def from_json(d: dict) -> "FileChange":
        return FileChange(
            path=d["path"],
            existed_before=bool(d["existed_before"]),
            before_content=d.get("before_content"),
        )


@dataclass
class Checkpoint:
    id: str
    created_at: float
    label: str
    changes: list[FileChange] = field(default_factory=list)
    skipped_large: list[str] = field(default_factory=list)
    skipped_binary: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "label": self.label,
            "changes": [c.to_json() for c in self.changes],
            "skipped_large": self.skipped_large,
            "skipped_binary": self.skipped_binary,
        }

    @staticmethod
    def from_json(d: dict) -> "Checkpoint":
        return Checkpoint(
            id=str(d["id"]),
            created_at=float(d["created_at"]),
            label=str(d.get("label", "")),
            changes=[FileChange.from_json(c) for c in d.get("changes", [])],
            skipped_large=list(d.get("skipped_large", [])),
            skipped_binary=list(d.get("skipped_binary", [])),
        )


def _workspace_dir(root: Path) -> Path:
    return CHECKPOINTS_ROOT / workspace_key(root)


class CheckpointStore:
    """Per-workspace stack of checkpoints, persisted so /undo survives restarts.

    Nothing is written to disk (not even an empty directory) until a turn
    actually mutates a file — merely opening a workspace shouldn't leave
    traces in ~/.tythancode.
    """

    def __init__(self, workspace_root: Path, storage_dir: Path | None = None):
        self.root = workspace_root.resolve()
        self.dir = storage_dir if storage_dir is not None else _workspace_dir(self.root)
        self._current: Checkpoint | None = None

    # -- recording (during a turn) ---------------------------------------

    def begin_turn(self, label: str) -> None:
        """Start collecting changes for a new turn. Call once per user turn."""
        self._current = Checkpoint(
            id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            label=" ".join(label.split())[:120],
        )

    def record_before(self, target: Path) -> None:
        """Capture `target`'s current (pre-mutation) content, once per turn per path.

        Only ever called for paths write_file/edit_file are *about* to touch —
        the actual write/edit may still go on to fail its own validation (e.g.
        `target` turns out to be a directory, or an edit's old_string doesn't
        match). This deliberately records nothing in that case rather than a
        checkpoint entry for a change that never happened.
        """
        if self._current is None:
            return
        key = str(target)
        if any(c.path == key for c in self._current.changes):
            return  # keep the *first* pre-turn state if the same file is touched twice
        if target.exists() and not target.is_file():
            return  # not a regular file (e.g. a directory) — write_file/edit_file will
            # reject this on their own; nothing meaningful to checkpoint or undo.
        existed = target.is_file()
        before: str | None
        if existed:
            try:
                if target.stat().st_size > MAX_CHECKPOINT_FILE_BYTES:
                    if key not in self._current.skipped_large:
                        self._current.skipped_large.append(key)
                    return
                raw = target.read_bytes()
            except OSError:
                return
            try:
                before = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Not valid UTF-8: read_text(errors="replace") would silently
                # corrupt the byte content before it's even stored, making
                # /undo "restore" a lossy, already-mangled version. Refuse to
                # checkpoint it instead — same honest-skip treatment as an
                # oversized file.
                if key not in self._current.skipped_binary:
                    self._current.skipped_binary.append(key)
                return
        else:
            before = None
        self._current.changes.append(FileChange(path=key, existed_before=existed, before_content=before))

    def commit_turn(self) -> Checkpoint | None:
        """Persist the turn's recorded changes as one checkpoint. Returns it, or
        None if the turn recorded nothing at all (no file was touched).

        A turn that touched only skipped (oversized/binary) files still comes
        back non-None, so the caller can tell the user those files aren't
        covered by /undo — but nothing is written to disk for it, since there
        is no actual undo-able state to persist or list in /checkpoints.
        """
        cp, self._current = self._current, None
        if cp is None or (not cp.changes and not cp.skipped_large and not cp.skipped_binary):
            return None
        if not cp.changes:
            return cp  # nothing to persist — just skipped-file bookkeeping to report
        self.dir.mkdir(parents=True, exist_ok=True)
        # A monotonic sequence number (not just the wall-clock timestamp) so
        # ordering stays correct even when two turns commit within the same
        # clock tick — filename sort must match commit order for /undo to
        # ever pop the right one.
        seq = self._next_sequence()
        path = self.dir / f"{seq:010d}_{cp.id}.json"
        path.write_text(json.dumps(cp.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._prune()
        return cp

    def _next_sequence(self) -> int:
        max_seq = -1
        for f in self._files():
            try:
                max_seq = max(max_seq, int(f.name.split("_", 1)[0]))
            except ValueError:
                continue
        return max_seq + 1

    # -- listing / undo ---------------------------------------------------

    def _files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted(self.dir.glob("*.json"))

    def _prune(self) -> None:
        files = self._files()
        excess = len(files) - MAX_CHECKPOINTS_PER_WORKSPACE
        for f in files[: max(excess, 0)]:
            f.unlink(missing_ok=True)

    def count(self) -> int:
        """Total number of checkpoints retained on disk (may exceed what list() returns)."""
        return len(self._files())

    def list(self, limit: int = 10) -> list[Checkpoint]:
        """Most recent first."""
        if limit <= 0:
            return []
        out = []
        for f in reversed(self._files()[-limit:]):
            try:
                out.append(Checkpoint.from_json(json.loads(f.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return out

    def undo_last(self) -> Checkpoint | None:
        """Pop and apply the most recent checkpoint. Returns it, or None if empty.

        Best-effort per file: a problem restoring one change (permission
        error, a path that no longer makes sense, a symlink now pointing
        somewhere unexpected) doesn't abort the rest of the checkpoint or
        crash the caller — every apply below and the checkpoint file itself
        are always cleaned up.
        """
        files = self._files()
        if not files:
            return None
        last = files[-1]
        try:
            cp = Checkpoint.from_json(json.loads(last.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            last.unlink(missing_ok=True)
            return None
        for change in cp.changes:
            try:
                # Re-resolve at undo time (not just the raw stored path) so a
                # symlink introduced after the checkpoint was recorded can't
                # smuggle the write outside the workspace.
                target = Path(change.path).resolve()
                if target != self.root and self.root not in target.parents:
                    continue  # refuse to touch anything outside this checkpoint's workspace
                if change.existed_before:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(change.before_content or "", encoding="utf-8")
                elif target.is_file():
                    target.unlink()
            except OSError:
                continue  # best-effort: skip this file, still process the rest
        last.unlink(missing_ok=True)
        return cp
