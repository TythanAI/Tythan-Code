"""Entry point: argument parsing and the interactive REPL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from . import __version__
from .agent import Agent
from .config import DEFAULT_EFFORT, Config, load_mcp_server_configs, load_provider_configs
from .mcp_client import MCPManager
from .providers import BackendConfigError, make_backend
from .tools import expand_mentions
from .ui import UI

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tythancode",
        description="A terminal AI coding assistant: chat with any model about your project, "
        "let it read/edit files, search code and run commands with your confirmation.",
    )
    parser.add_argument("workspace", nargs="?", default=".", help="project directory (default: current)")
    parser.add_argument("--provider", help="provider name from ~/.tythancode/config.json")
    parser.add_argument("--model", help="override the provider's model id")
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORT_LEVELS,
                        help="reasoning effort (Anthropic models)")
    parser.add_argument("--yolo", action="store_true", help="skip confirmation prompts (dangerous)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="don't record file checkpoints (disables /undo)")
    parser.add_argument("-p", "--prompt", help="run a single prompt non-interactively and exit")
    parser.add_argument("--version", action="version", version=f"Tythan Code {__version__}")
    return parser.parse_args(argv)


def build_backend(name: str, providers: dict, config: Config, model_override: str | None = None):
    if name not in providers:
        raise BackendConfigError(
            f"unknown provider '{name}' — available: {', '.join(providers)} "
            "(edit ~/.tythancode/config.json to add more)"
        )
    backend = make_backend(providers[name], config)
    if model_override:
        backend.model = model_override
    return backend


def connect_mcp_servers(ui: UI) -> MCPManager:
    """Load and connect every MCP server from ~/.tythancode/config.json.
    Never raises: a server that fails to start (or a config typo) is
    reported and simply contributes no tools, same as a provider outage
    shouldn't block the rest of Tythan Code from working."""
    configs, errors = load_mcp_server_configs()
    for err in errors:
        ui.error(err)
    manager = MCPManager(configs)
    if not configs:
        return manager
    manager.connect_all()
    if manager.connected_servers:
        tool_count = len(manager.tools)
        ui.info(f"MCP: connected to {', '.join(manager.connected_servers)} ({tool_count} tool(s))")
    for name, err in manager.startup_errors.items():
        ui.error(f"MCP server '{name}' failed to start: {err}")
    return manager


def handle_slash(command: str, agent: Agent, ui: UI, providers: dict) -> bool:
    """Handle a /command. Returns False when the REPL should exit."""
    parts = command.split(maxsplit=1)
    name, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")
    if name in ("/exit", "/quit"):
        return False
    if name == "/help":
        ui.help()
    elif name == "/clear":
        agent.reset()
        ui.info("conversation cleared")
    elif name == "/provider":
        if not arg:
            current = agent.backend.name
            listing = ", ".join(f"[bold]{n}[/bold]" if n == current else n for n in providers)
            ui.info(f"providers: {listing} — switch with /provider <name>")
        else:
            try:
                agent.set_backend(build_backend(arg, providers, agent.config))
                ui.info(f"switched to {agent.backend.describe()} (conversation reset)")
            except BackendConfigError as exc:
                ui.error(str(exc))
    elif name == "/model":
        if arg:
            agent.backend.model = arg
            ui.info(
                f"model set to {arg} — context window stays at {agent.backend.context_window} "
                "(set context_window for this provider in ~/.tythancode/config.json if the new "
                "model's real window differs; check with /context)"
            )
        else:
            ui.info(f"current: {agent.backend.describe()}")
    elif name == "/effort":
        if arg in EFFORT_LEVELS:
            agent.config.effort = arg
            ui.info(f"effort set to {arg} (applies to Anthropic models)")
        else:
            ui.info(f"current effort: {agent.config.effort} (choose from {', '.join(EFFORT_LEVELS)})")
    elif name == "/audit":
        from .sca import scan_dependencies
        from .security import scan_workspace

        findings = scan_workspace(agent.workspace, arg or ".")
        dep_findings, note = scan_dependencies(agent.workspace.root)
        ui.audit_report(findings + dep_findings)
        ui.info(note)
        if findings or dep_findings:
            ui.info('ask the assistant to "fix the audit findings" to remediate them')
    elif name == "/yolo":
        agent.config.yolo = not agent.config.yolo
        ui.info("confirmations OFF — all actions auto-approved" if agent.config.yolo else "confirmations ON")
    elif name == "/compact":
        if not agent.maybe_compact(force=True):
            ui.info("nothing worth compacting yet (not enough history)")
    elif name == "/context":
        used = agent.context_tokens_estimate()
        budget = agent.token_budget()
        pct = int(100 * used / budget) if budget else 0
        ui.info(
            f"context: ~{used} / {budget} tokens in use ({pct}%) — "
            f"window {agent.backend.context_window}, {agent.config.max_tokens} reserved for output"
        )
    elif name == "/undo":
        checkpoint = agent.checkpoints.undo_last()
        if checkpoint is None:
            ui.info("nothing to undo")
        else:
            skipped = checkpoint.skipped_large + checkpoint.skipped_binary
            note = f" (note: {len(skipped)} large/binary file(s) in this checkpoint weren't covered)" if skipped else ""
            ui.info(f'reverted {len(checkpoint.changes)} file(s) from "{checkpoint.label}"{note}')
    elif name == "/todos":
        ui.todo_list(agent.todos)
    elif name == "/reindex":
        index = agent.reindex()
        ui.info(f"indexed {index.files_indexed} file(s), {len(index.chunks)} chunk(s) for code_search")
    elif name == "/mcp":
        ui.mcp_status(agent.mcp)
    elif name == "/checkpoints":
        try:
            limit = int(arg) if arg else 10
        except ValueError:
            limit = 10
        ui.checkpoints_list(agent.checkpoints.list(limit=limit), total=agent.checkpoints.count())
    else:
        ui.info(f"unknown command: {name} (try /help)")
    return True


def run_turn_safely(agent: Agent, ui: UI, text: str) -> None:
    try:
        # Pass the raw text as the checkpoint label — expand_mentions() inlines
        # full file contents into what the model sees, which would otherwise
        # end up as unreadable noise in /checkpoints.
        agent.run_turn(expand_mentions(text, agent.workspace), label=text)
    except anthropic.AuthenticationError:
        ui.error(
            "authentication failed — set ANTHROPIC_API_KEY (or log in with `ant auth login`) and retry"
        )
    except anthropic.RateLimitError:
        ui.error("rate limited — wait a moment and try again")
    except anthropic.APIConnectionError:
        ui.error("network error — check your connection and try again")
    except anthropic.APIStatusError as exc:
        ui.error(f"API error {exc.status_code}: {exc.message}")
    except TypeError as exc:
        # The Anthropic SDK raises TypeError when no credentials can be resolved.
        if "authentication" in str(exc).lower():
            ui.error(
                "no API credentials found — set ANTHROPIC_API_KEY (or log in with `ant auth login`)"
            )
        else:
            raise
    except KeyboardInterrupt:
        ui.flush_stream()
        ui.info("\ninterrupted — the partial turn was kept in history")
    except Exception as exc:  # OpenAI-compatible providers raise their own classes
        ui.error(f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"error: workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    config = Config(workspace=workspace, effort=args.effort, yolo=args.yolo,
                    checkpoints_enabled=not args.no_checkpoints)
    ui = UI()

    try:
        default_provider, providers = load_provider_configs()
        backend = build_backend(args.provider or default_provider, providers, config, args.model)
    except (BackendConfigError, ValueError) as exc:
        ui.error(str(exc))
        return 2

    mcp_manager = connect_mcp_servers(ui)
    agent = Agent(config, ui, backend, mcp_manager=mcp_manager)

    try:
        if args.prompt:
            run_turn_safely(agent, ui, args.prompt)
            return 0

        ui.banner(workspace, backend.describe(), config.effort, config.yolo,
                  rules_source=agent.project_rules.source if agent.project_rules else None)
        session: PromptSession = PromptSession(
            history=FileHistory(str(Path.home() / ".tythancode_history"))
        )

        while True:
            try:
                text = session.prompt("you> ").strip()
            except KeyboardInterrupt:
                continue  # clear the current input line
            except EOFError:
                break
            if not text:
                continue
            if text.startswith("/"):
                try:
                    if not handle_slash(text, agent, ui, providers):
                        break
                except KeyboardInterrupt:
                    ui.info("\ninterrupted")
                except Exception as exc:
                    # A slash command failing (e.g. /undo hitting a disk error) must
                    # drop back to the prompt, not take the whole session down —
                    # unlike run_turn_safely's LLM-turn path, these have no reason
                    # to ever propagate.
                    ui.error(f"{type(exc).__name__}: {exc}")
                continue
            run_turn_safely(agent, ui, text)

        ui.info("bye!")
        return 0
    finally:
        # Always tear down MCP server subprocesses on the way out, whether
        # we're exiting normally, via /exit, or an exception is propagating.
        agent.mcp.close()


if __name__ == "__main__":
    sys.exit(main())
