"""Git-branch isolation for unattended ("background agent") runs.

`--branch` checks out a fresh branch before the run starts and commits
whatever's left uncommitted once it ends, so a headless `-p` run (or a long
interactive session someone wants isolated) doesn't land its changes on
whatever branch happened to be checked out when it started.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .ui import UI


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def is_git_repo(workspace: Path) -> bool:
    return _run_git(["rev-parse", "--is-inside-work-tree"], workspace).returncode == 0


def slugify(text: str, max_len: int = 40) -> str:
    """Turn free text into a branch-name-safe slug; never returns empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "task"


def start_branch(workspace: Path, name: str | None, prompt: str, ui: UI) -> str | None:
    """Create and check out a new branch for an isolated automated run.

    Returns the branch name actually checked out, or None if the workspace
    isn't a git repository or branch creation failed for any other reason —
    in which case the run proceeds on whatever's currently checked out, same
    as if `--branch` hadn't been passed at all, rather than aborting a task
    the user asked for over a convenience feature.
    """
    if not is_git_repo(workspace):
        ui.error("--branch needs a git repository here — continuing without branch isolation")
        return None
    branch = name or f"tythancode/{slugify(prompt)}-{int(time.time())}"
    result = _run_git(["checkout", "-b", branch], workspace)
    if result.returncode != 0:
        ui.error(f"couldn't create branch '{branch}': {result.stderr.strip()}")
        return None
    ui.info(f"working on new branch: {branch}")
    return branch


def commit_leftover_changes(workspace: Path, branch: str, ui: UI) -> None:
    """Commit whatever's left uncommitted at the end of a --branch run.

    A no-op if there's nothing to commit — e.g. the agent already committed
    its own work via run_command, or the run never actually changed
    anything.
    """
    status = _run_git(["status", "--porcelain"], workspace)
    if not status.stdout.strip():
        return
    _run_git(["add", "-A"], workspace)
    commit = _run_git(["commit", "-m", f"tythancode: automated changes on {branch}"], workspace)
    if commit.returncode == 0:
        ui.info(f"committed remaining changes on {branch}")
    else:
        ui.error(f"couldn't commit changes on {branch}: {commit.stderr.strip()}")
