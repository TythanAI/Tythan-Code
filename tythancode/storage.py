"""Shared helper for per-workspace storage directories under ~/.tythancode/.

Checkpoints and the code_search index cache each need their own directory
per project, keyed off the workspace path so unrelated projects never
collide. One function, used by both, so the key is computed the same way
everywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def workspace_key(root: Path) -> str:
    """Stable, filesystem-safe identifier for a workspace path."""
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
