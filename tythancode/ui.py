"""Terminal rendering: banner, streamed text, tool calls, diffs, confirmations."""

from __future__ import annotations

import difflib
import json
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
            while True:
                answer = self.console.input(
                    "[bold]apply this hunk?[/bold] [dim]\\[y/n/a/d/?][/dim] "
                ).strip().lower()
                if answer in ("y", "yes", "д", "да"):
                    accepted.append(True)
                    break
                if answer in ("n", "no", "н", "нет", ""):
                    accepted.append(False)
                    break
                if answer == "a":
                    accepted.append(True)
                    apply_rest = True
                    break
                if answer == "d":
                    accepted.append(False)
                    apply_rest = False
                    break
                self.console.print(
                    "[dim]y = apply this hunk, n = skip it, "
                    "a = apply this and every remaining hunk, "
                    "d = skip this and every remaining hunk[/dim]"
                )
        return accepted

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
