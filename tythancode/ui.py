"""Terminal rendering: banner, streamed text, tool calls, diffs, confirmations."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

BANNER = r"""
 _        _   _                           _
| |_ _  _| |_| |_  __ _ _ _    __ ___  __| |___
|  _| || |  _| ' \/ _` | ' \  / _/ _ \/ _` / -_)
 \__|\_, |\__|_||_\__,_|_||_| \__\___/\__,_\___|
     |__/
"""


@dataclass
class FileReview:
    """One file's pending write_file/edit_file change, ready for
    `UI.review_batch` — the diff is already computed, nothing has touched
    disk yet. `hunks` is unused (empty) when `is_new`, since a brand-new
    file gets a single whole-file diff instead of per-hunk review."""

    call_id: str
    path: str
    is_new: bool
    hunks: list  # list[Hunk]
    new_content: str = ""  # only rendered when is_new


class UI:
    def __init__(self, console: Console | None = None):
        self.console = console or Console(highlight=False)
        self._streamed_text: list[str] = []

    # -- chrome --------------------------------------------------------

    def banner(self, workspace: Path, backend: str, effort: str, yolo: bool, rules_source: str | None = None) -> None:
        self.console.print(Text(BANNER, style="bold cyan"))
        self.console.print(f"[dim]workspace:[/dim] [bold]{workspace}[/bold]")
        mode = "[red]yolo (no confirmations!)[/red]" if yolo else "confirm writes & commands"
        self.console.print(f"[dim]backend:[/dim] {backend}  [dim]effort:[/dim] {effort}  [dim]mode:[/dim] {mode}")
        if rules_source:
            self.console.print(f"[dim]project instructions:[/dim] loaded from {rules_source}")
        self.console.print("[dim]/help for commands, Ctrl+D to exit[/dim]\n")

    def help(self) -> None:
        self.console.print(
            Panel(
                "\n".join(
                    [
                        "[bold]/help[/bold]            show this help",
                        "[bold]/clear[/bold]           reset the conversation",
                        "[bold]/provider \\[name][/bold] list providers or switch (resets the chat)",
                        "[bold]/model <id>[/bold]      switch model within the current provider",
                        "[bold]/effort <lvl>[/bold]    low | medium | high | xhigh | max (Anthropic)",
                        "[bold]/audit \\[path][/bold]    security scan: secrets, dangerous code, bad config",
                        "[bold]/yolo[/bold]            toggle confirmation prompts",
                        "[bold]/todos[/bold]           show the assistant's current task plan",
                        "[bold]/reindex[/bold]         rebuild the code_search index now",
                        "[bold]/mcp[/bold]             show connected MCP servers and their tools",
                        "[bold]/undo[/bold]            revert the file changes from the last turn",
                        "[bold]/checkpoints[/bold]     list recent undo checkpoints",
                        "[bold]/compact[/bold]         summarize older history now to free up context",
                        "[bold]/context[/bold]         show estimated context usage",
                        "[bold]/exit[/bold]            quit (also Ctrl+D)",
                        "",
                        "Anything else is sent to the assistant. It can read/edit files,",
                        "search the project and run commands — mutations ask for your OK first.",
                        "Mention files with @path/to/file to attach their contents.",
                        "Providers are configured in ~/.tythancode/config.json.",
                    ]
                ),
                title="Tythan Code",
                border_style="cyan",
            )
        )

    def info(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")

    def error(self, message: str) -> None:
        self.console.print(f"[bold red]error:[/bold red] {message}")

    # -- streaming -----------------------------------------------------

    def assistant_prefix(self) -> None:
        self.console.print("[bold magenta]assistant[/bold magenta]")
        self._streamed_text = []

    def stream_text(self, chunk: str) -> None:
        # Stream raw for immediacy; the final markdown re-render happens in flush.
        self._streamed_text.append(chunk)
        self.console.print(chunk, end="", soft_wrap=True)

    def thinking_started(self) -> None:
        self.console.print("[dim italic]thinking…[/dim italic]")

    def flush_stream(self) -> None:
        if self._streamed_text:
            self.console.print()  # newline after raw stream
        self._streamed_text = []

    def render_final_text(self, text: str) -> None:
        """Optional pretty re-render of the full reply as markdown."""
        if text.strip():
            self.console.print(Markdown(text))

    # -- tools ---------------------------------------------------------

    def tool_call(self, name: str, tool_input: dict) -> None:
        preview = json.dumps(tool_input, ensure_ascii=False)
        if len(preview) > 200:
            preview = preview[:200] + "…"
        self.console.print(f"\n[bold yellow]⚙ {name}[/bold yellow] [dim]{preview}[/dim]")

    def tool_result(self, output: str, is_error: bool = False) -> None:
        style = "red" if is_error else "dim"
        lines = output.splitlines() or [""]
        shown = "\n".join(lines[:12])
        if len(lines) > 12:
            shown += f"\n… ({len(lines) - 12} more lines)"
        self.console.print(Text(shown, style=style))

    def show_diff(self, path: str, old: str, new: str) -> None:
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        body = Text()
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                body.append(line, style="green")
            elif line.startswith("-") and not line.startswith("---"):
                body.append(line, style="red")
            elif line.startswith("@@"):
                body.append(line, style="cyan")
            else:
                body.append(line, style="dim")
        self.console.print(Panel(body, title=f"changes to {path}", border_style="yellow"))

    def _hunk_panel(self, path: str, hunk, index: int, total: int) -> Panel:
        header = f"@@ -{hunk.old_start},{len(hunk.old_lines)} +{hunk.new_start},{len(hunk.new_lines)} @@"
        body = Text()
        body.append(header + "\n", style="cyan")
        for line in hunk.old_lines:
            body.append(f"-{line.rstrip(chr(10))}\n", style="red")
        for line in hunk.new_lines:
            body.append(f"+{line.rstrip(chr(10))}\n", style="green")
        return Panel(body, title=f"{path} — hunk {index}/{total}", border_style="yellow")

    def _prompt_apply_choice(self, prompt: str) -> str:
        """Prompt for y/n/a/d, reprompting on anything else. Returns the
        normalized single-letter choice: 'y', 'n', 'a' or 'd'."""
        while True:
            answer = self.console.input(
                f"[bold]{prompt}[/bold] [dim]\\[y/n/a/d/?][/dim] "
            ).strip().lower()
            if answer in ("y", "yes", "д", "да"):
                return "y"
            if answer in ("n", "no", "н", "нет", ""):
                return "n"
            if answer in ("a", "d"):
                return answer
            self.console.print(
                "[dim]y = apply this change, n = skip it, "
                "a = apply this and everything remaining, "
                "d = skip this and everything remaining[/dim]"
            )

    def review_hunks(self, path: str, hunks: list) -> list[bool]:
        """Ask the user to accept or reject each hunk independently, git
        add -p style. Returns one bool per hunk, same order as `hunks`."""
        accepted: list[bool] = []
        apply_rest: bool | None = None  # once set, answers every remaining hunk
        total = len(hunks)
        for i, hunk in enumerate(hunks, start=1):
            if apply_rest is not None:
                accepted.append(apply_rest)
                continue
            self.console.print(self._hunk_panel(path, hunk, i, total))
            choice = self._prompt_apply_choice("apply this hunk?")
            if choice == "a":
                apply_rest = True
            elif choice == "d":
                apply_rest = False
            accepted.append(choice in ("y", "a"))
        return accepted

    def _new_file_panel(self, path: str, new_text: str) -> Panel:
        diff = difflib.unified_diff(
            [], new_text.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{path}"
        )
        body = Text()
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                body.append(line, style="green")
            elif line.startswith("@@"):
                body.append(line, style="cyan")
            else:
                body.append(line, style="dim")
        return Panel(body, title=f"{path} — new file", border_style="yellow")

    def review_batch(self, reviews: list[FileReview]) -> dict[str, list[bool]]:
        """Review every file's pending change as one continuous accept/reject
        stream instead of one file at a time: a summary panel first (when
        more than one file is involved), then each file's hunks — or, for a
        new file, its single whole-file diff — in order. y/n/a/d, same as
        `review_hunks`, except 'a'/'d' apply to everything remaining in the
        whole batch, not just the current file.

        Returns {call_id: [accepted...]}, aligned with each review's hunks
        (a new file's entry always has exactly one element, since it's
        accepted or declined as a whole)."""
        if not reviews:
            return {}
        if len(reviews) > 1:
            lines = "\n".join(
                f"{r.path}  ({'new file' if r.is_new else f'{len(r.hunks)} hunk(s)'})"
                for r in reviews
            )
            self.console.print(
                Panel(lines, title=f"{len(reviews)} file(s) will change", border_style="yellow")
            )

        decisions: dict[str, list[bool]] = {}
        apply_rest: bool | None = None
        for review in reviews:
            if review.is_new:
                if apply_rest is None:
                    self.console.print(self._new_file_panel(review.path, review.new_content))
                    choice = self._prompt_apply_choice("apply this change?")
                    if choice == "a":
                        apply_rest = True
                    elif choice == "d":
                        apply_rest = False
                    decisions[review.call_id] = [choice in ("y", "a")]
                else:
                    decisions[review.call_id] = [apply_rest]
                continue

            accepted: list[bool] = []
            total = len(review.hunks)
            for i, hunk in enumerate(review.hunks, start=1):
                if apply_rest is not None:
                    accepted.append(apply_rest)
                    continue
                self.console.print(self._hunk_panel(review.path, hunk, i, total))
                choice = self._prompt_apply_choice("apply this change?")
                if choice == "a":
                    apply_rest = True
                elif choice == "d":
                    apply_rest = False
                accepted.append(choice in ("y", "a"))
            decisions[review.call_id] = accepted
        return decisions

    def audit_report(self, findings: list) -> None:
        if not findings:
            self.console.print("[bold green]✓ no security findings[/bold green] "
                               "[dim](secrets, dangerous patterns, insecure config)[/dim]")
            return
        style_for = {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow"}
        body = Text()
        for f in findings:
            body.append(f"{f.severity:<9}", style=style_for.get(f.severity, "white"))
            body.append(f"{f.path}:{f.line}  ", style="bold")
            body.append(f"{f.message}\n")
            body.append(f"          {f.snippet}\n", style="dim")
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        title = f"security audit — {len(findings)} finding(s) ({', '.join(f'{k} {v}' for k, v in counts.items())})"
        self.console.print(Panel(body, title=title, border_style="red"))

    def todo_list(self, todos: list[dict]) -> None:
        if not todos:
            self.console.print("[dim]todo list cleared[/dim]")
            return
        icons = {"completed": ("✓", "green"), "in_progress": ("→", "bold yellow"), "pending": ("○", "dim")}
        body = Text()
        for t in todos:
            icon, style = icons.get(t["status"], ("?", "white"))
            content_style = "dim strike" if t["status"] == "completed" else style
            body.append(f"{icon} ", style=style)
            body.append(f"{t['content']}\n", style=content_style)
        done = sum(1 for t in todos if t["status"] == "completed")
        self.console.print(Panel(body, title=f"plan ({done}/{len(todos)} done)", border_style="cyan"))

    def mcp_status(self, manager) -> None:
        if not manager.configs:
            self.console.print("[dim]no MCP servers configured — add one under \"mcp_servers\" in "
                               "~/.tythancode/config.json[/dim]")
            return
        body = Text()
        for cfg in manager.configs:
            if cfg.name in manager.startup_errors:
                body.append(f"✗ {cfg.name}  ", style="bold red")
                body.append(f"{manager.startup_errors[cfg.name]}\n", style="dim")
                continue
            body.append(f"✓ {cfg.name}\n", style="bold green")
            for tool in manager.tools:
                if tool.server == cfg.name:
                    hint = " (read-only)" if tool.read_only else ""
                    body.append(f"    {tool.qualified_name}{hint}\n", style="dim")
        self.console.print(Panel(body, title="MCP servers", border_style="cyan"))

    def confirm(self, prompt: str) -> bool:
        answer = self.console.input(f"[bold]{prompt}[/bold] [dim]\\[y/N][/dim] ")
        return answer.strip().lower() in ("y", "yes", "д", "да")

    def checkpoints_list(self, checkpoints: list, total: int | None = None) -> None:
        if not checkpoints:
            self.console.print("[dim]no checkpoints yet — checkpoints are created when the "
                               "assistant writes or edits a file[/dim]")
            return
        import datetime

        body = Text()
        for i, cp in enumerate(checkpoints):
            ts = datetime.datetime.fromtimestamp(cp.created_at).strftime("%H:%M:%S")
            marker = " (latest — /undo reverts this one)" if i == 0 else ""
            body.append(f"{ts}  ", style="dim")
            body.append(f"{len(cp.changes)} file(s)  ", style="bold")
            body.append(f"{cp.label}{marker}\n")
        title = f"checkpoints ({len(checkpoints)})"
        if total is not None and total > len(checkpoints):
            title = f"checkpoints (showing {len(checkpoints)} of {total} — /checkpoints <n> to see more)"
        self.console.print(Panel(body, title=title, border_style="cyan"))
