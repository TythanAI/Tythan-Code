"""The agent loop: stream a response, execute requested tools, repeat.

Provider-agnostic: all model I/O goes through a Backend, which owns the
native message format. The agent owns tool execution and user confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoints import CheckpointStore
from .codeindex import CodeIndex, build_index, format_hits, search_index
from .compaction import cap_head, estimate_tokens_heuristic, split_into_rounds
from .config import Config
from .hunks import Hunk, apply_selected_hunks, split_hunks
from .mcp_client import MCPManager
from .providers.base import Backend, ToolCall, ToolResult
from .rules import load_project_rules
from .storage import workspace_key
from .tools import TOOL_DEFINITIONS, ToolError, Workspace, validate_todos
from .ui import UI, FileReview

DECLINED = "The user declined this action. Ask them how to proceed or try another approach."

# write_file/edit_file calls in the same round are batched together for
# review (see Agent._review_file_batch) instead of being confirmed one at a
# time; every other mutating tool (run_command, fetch_url, ...) keeps its
# own plain yes/no at its original position in the round.
MUTATING_FILE_TOOLS = frozenset({"write_file", "edit_file"})


@dataclass
class PendingChange:
    """A prepared-but-not-yet-approved write_file/edit_file call: the diff
    against what's on disk is already computed so it can be reviewed, but
    nothing has been written yet."""

    call_id: str
    path: str
    old_content: str
    new_content: str
    hunks: list[Hunk]  # split_hunks(old_content, new_content); empty for a new file
    is_new: bool  # True when old_content is empty — nothing to diff hunk-by-hunk against


# Persistent code_search index cache, one subdirectory per workspace (see
# storage.workspace_key). Overridable per-Agent for tests, so pytest never
# touches the real ~/.tythancode.
DEFAULT_INDEX_CACHE_ROOT = Path.home() / ".tythancode" / "index"

# Compact when the estimated context in use crosses this fraction of the
# token budget (context window minus the reserved output tokens). Left with
# real headroom below 1.0 because the estimate can be approximate (the
# character-based heuristic fallback) and providers vary in exactly how they
# count tokens.
COMPACT_TRIGGER_RATIO = 0.8

# Floor for the token budget so a tiny/misconfigured context_window can't make
# every single call trigger compaction.
MIN_TOKEN_BUDGET = 1000

# Cap on how much of the old-rounds transcript is fed to the summarization
# call, so compaction itself can't blow up the context it's trying to shrink.
MAX_SUMMARY_INPUT_CHARS = 60_000

SUMMARY_PROMPT = """\
Summarize the earlier part of this coding session so the assistant can keep \
working with full context after older messages are dropped. Be concrete and \
specific:
- what the user has asked for, across all their messages so far
- what has been done in response (files read, files changed and how, \
commands run and their outcome)
- open problems, errors seen, or things still left to do
- any project-specific facts learned along the way (conventions, file \
locations, decisions made, things that didn't work)

Skip pleasantries and internal reasoning. Write it as plain prose working \
memory for the assistant to keep using, not a transcript. Be thorough about \
facts and decisions, but don't pad it out.
"""

SYSTEM_PROMPT = """\
You are Tythan Code, an AI coding assistant running in the user's terminal.
You operate on the user's project workspace via tools: read_file, write_file,
edit_file, list_files, search, code_search, fetch_url, run_command. Additional
`mcp__<server>__<tool>` tools may also be available, connected from the
user's own MCP server configuration — use them the same way when relevant.

Guidelines:
- Explore before you change: read the relevant files first so edits match the
  existing code style and edit_file old_string matches exactly.
- Use `search` for an exact string or regex you already know. Use
  `code_search` instead when you only know roughly what you're looking for
  (e.g. "where is retry logic for HTTP calls") — it ranks chunks by keyword
  relevance rather than requiring an exact match.
- Prefer edit_file for small changes; write_file for new files or rewrites.
  Always output complete file contents in write_file — never placeholders.
- When a change spans several files and you already know what every one of
  them needs (e.g. renaming something and updating each call site, or
  threading a new field through the places that use it), call
  write_file/edit_file for all of them in the same response instead of one
  at a time — the user reviews the whole batch together in one pass instead
  of being blocked on each file in turn. Only spread edits across separate
  responses when a later file genuinely depends on something you'll only
  know afterward — a command's output, or a file you haven't read yet.
- Mutating actions (writes, edits, commands) are shown to the user for
  confirmation; a denied action means the user declined it, so adjust your
  approach instead of retrying the same call.
- After making changes, verify them when practical (run tests, run the code).
- For multi-step work (roughly 3+ distinct steps), call todo_write with the
  full plan before starting, keep exactly one item in_progress at a time, and
  mark items completed as soon as they're actually done. Skip it for small or
  single-step requests — it's for keeping the user oriented on longer tasks,
  not overhead on every message.
- Security first: after writing or significantly changing code, run
  security_scan on the touched files and fix CRITICAL/HIGH findings before
  declaring the task done. Never hardcode secrets; read them from env vars.
- Keep answers concise and grounded in what you actually observed in the
  workspace. Lead with the outcome.
"""


class Agent:
    def __init__(
        self,
        config: Config,
        ui: UI,
        backend: Backend,
        checkpoint_store: CheckpointStore | None = None,
        mcp_manager: MCPManager | None = None,
        code_index_cache_dir: Path | None = None,
    ):
        self.config = config
        self.ui = ui
        self.backend = backend
        self.workspace = Workspace(config.workspace)
        self._code_index_cache_dir = (
            code_index_cache_dir if code_index_cache_dir is not None
            else DEFAULT_INDEX_CACHE_ROOT / workspace_key(self.workspace.root)
        )
        # An empty manager if the caller didn't wire one up (or no MCP
        # servers are configured): every lookup finds no tools and
        # connect_all()/close() are no-ops, so behavior is identical to
        # Tythan Code without MCP support.
        self.mcp = mcp_manager if mcp_manager is not None else MCPManager([])
        # Snapshotted once, same tradeoff as workspace.ignore: a rules file
        # edited mid-session takes effect on the next restart, not instantly.
        self.project_rules = load_project_rules(self.workspace.root)
        self.messages: list = []
        self.todos: list[dict] = []
        # Lazy — built on first code_search call, not on startup, so opening
        # Tythan Code for a quick one-off task doesn't pay an indexing cost
        # it'll never use. Dropped (see _execute_tool) after any tool call
        # that could have changed files, so a stale index never silently
        # serves outdated results; when it does rebuild, the on-disk cache
        # (see codeindex.build_index) skips re-tokenizing anything whose
        # mtime/size didn't actually change.
        self._code_index: CodeIndex | None = None
        self.checkpoints = checkpoint_store if checkpoint_store is not None else CheckpointStore(self.workspace.root)
        # Once a compaction attempt fails (e.g. network error during the
        # summarization call), stop retrying it every tool round of the
        # current turn — the underlying call is likely to keep failing, and
        # retrying costs a real API round trip each time.
        self._compaction_unavailable = False

    def reset(self) -> None:
        self.messages = []
        self.todos = []

    def set_backend(self, backend: Backend) -> None:
        """Switch provider. History is provider-native, so the conversation resets."""
        self.backend = backend
        self.reset()

    def system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT + f"\nWorkspace root: {self.workspace.root}"
        if self.project_rules is not None:
            prompt += (
                f"\n\nProject-specific instructions (from {self.project_rules.source}, "
                f"provided by the project — follow these unless they conflict with the "
                f"guidelines above):\n{self.project_rules.text}"
            )
        return prompt

    # -- tool dispatch ---------------------------------------------------

    def tool_definitions(self) -> list[dict]:
        """Built-in tools plus whatever the connected MCP servers expose,
        in the same {name, description, input_schema} shape either way."""
        return TOOL_DEFINITIONS + [t.to_definition() for t in self.mcp.tools]

    def _execute_tool(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """Run one tool. Returns (output, is_error)."""
        ws = self.workspace
        try:
            if name.startswith("mcp__"):
                return self._execute_mcp_tool(name, tool_input)

            if name == "read_file":
                return ws.read_file(
                    tool_input["path"],
                    offset=tool_input.get("offset", 1),
                    limit=tool_input.get("limit", 2000),
                ), False
            if name == "list_files":
                return ws.list_files(tool_input.get("pattern", "**/*")), False
            if name == "search":
                return ws.search(tool_input["pattern"], tool_input.get("glob", "**/*")), False
            if name == "write_file":
                _, old = ws.prepare_write(tool_input["path"], tool_input["content"])
                return self._review_and_write(tool_input["path"], old, tool_input["content"])
            if name == "edit_file":
                _, old, new = ws.prepare_edit(
                    tool_input["path"],
                    tool_input["old_string"],
                    tool_input["new_string"],
                    tool_input.get("replace_all", False),
                )
                return self._review_and_write(tool_input["path"], old, new)
            if name == "code_search":
                try:
                    top_k = max(1, min(int(tool_input.get("top_k", 8)), 20))
                except (TypeError, ValueError):
                    top_k = 8
                report = format_hits(search_index(self._get_code_index(), tool_input["query"], top_k=top_k))
                if self._code_index.files_skipped_limit:
                    report += (
                        f"\n\n[index covers the first {self._code_index.files_indexed} files "
                        "found — this project is large enough that results may be partial]"
                    )
                return report, False
            if name == "security_scan":
                from .security import format_findings, scan_workspace

                findings = scan_workspace(ws, tool_input.get("path", "."))
                report = format_findings(findings)
                if tool_input.get("include_dependencies"):
                    from .sca import scan_dependencies

                    dep_findings, note = scan_dependencies(ws.root)
                    report += f"\n\nDependencies: {note}"
                    if dep_findings:
                        report += "\n" + "\n".join(
                            f"[{f.severity}] {f.rule} {f.path} — {f.message}" for f in dep_findings
                        )
                return report, False
            if name == "todo_write":
                self.todos = validate_todos(tool_input.get("todos", []))
                self.ui.todo_list(self.todos)
                return f"Todo list updated ({len(self.todos)} item(s))", False
            if name == "fetch_url":
                if not self.config.yolo and not self.ui.confirm(f"fetch: {tool_input['url']} ?"):
                    return DECLINED, True
                from .webfetch import FetchError, fetch_url

                try:
                    return fetch_url(tool_input["url"]), False
                except FetchError as exc:
                    return str(exc), True
            if name == "run_command":
                if not self.config.yolo and not self.ui.confirm(f"run: {tool_input['command']} ?"):
                    return DECLINED, True
                result = ws.run_command(tool_input["command"])
                self._code_index = None  # the command may have changed files
                return result, False
            return f"Unknown tool: {name}", True
        except ToolError as exc:
            return str(exc), True
        except KeyError as exc:
            return f"Missing required parameter: {exc}", True

    def _review_and_write(self, path: str, old_content: str, new_content: str) -> tuple[str, bool]:
        """Shared path for write_file/edit_file: checkpoint the pre-change
        content, get approval, and write only what was approved.

        A file that doesn't exist yet gets a single whole-file confirm —
        there's nothing meaningful to split into hunks. An existing file
        gets per-hunk review (git add -p style) instead of an all-or-nothing
        yes/no, so the user can keep part of a multi-part change and reject
        the rest without declining the whole edit.

        Checkpointing only happens once a write is actually about to
        happen — not up front — so a fully declined change (nothing
        approved) leaves no checkpoint behind, same as before this
        approval path could partially apply a change.
        """
        if self.config.yolo:
            return self._commit(path, new_content), False

        if not old_content:
            self.ui.show_diff(path, old_content, new_content)
            if not self.ui.confirm(f"apply changes to {path}?"):
                return DECLINED, True
            return self._commit(path, new_content), False

        hunks = split_hunks(old_content, new_content)
        if not hunks:
            # New content identical to what's on disk — nothing to review.
            return self._commit(path, new_content), False

        accepted = self.ui.review_hunks(path, hunks)
        if not any(accepted):
            return DECLINED, True

        final_content = apply_selected_hunks(old_content, new_content, accepted)
        output = self._commit(path, final_content)
        if not all(accepted):
            output += f" ({sum(accepted)}/{len(hunks)} hunk(s) applied — the rest were declined)"
        return output, False

    def _prepare_pending_change(self, call: ToolCall) -> PendingChange:
        """Compute a write_file/edit_file call's diff against disk without
        writing anything, so a whole round's worth of them can be reviewed
        together before any of them actually commit. Raises ToolError/
        KeyError exactly like the underlying tool would on invalid input."""
        ws = self.workspace
        path = call.input["path"]
        if call.name == "write_file":
            _, old = ws.prepare_write(path, call.input["content"])
            new = call.input["content"]
        else:
            _, old, new = ws.prepare_edit(
                path, call.input["old_string"], call.input["new_string"],
                call.input.get("replace_all", False),
            )
        if not old:
            return PendingChange(call.id, path, old, new, hunks=[], is_new=True)
        return PendingChange(call.id, path, old, new, hunks=split_hunks(old, new), is_new=False)

    def _finalize_pending_change(self, change: PendingChange, accepted: list[bool] | None) -> tuple[str, bool]:
        """Apply a batch-reviewed change now that a decision has been made.
        `accepted` is None for a change with nothing to decide (new content
        identical to what's already on disk)."""
        if accepted is None:
            return self._commit(change.path, change.new_content), False
        if not any(accepted):
            return DECLINED, True
        if change.is_new:
            return self._commit(change.path, change.new_content), False
        final_content = apply_selected_hunks(change.old_content, change.new_content, accepted)
        output = self._commit(change.path, final_content)
        if not all(accepted):
            output += f" ({sum(accepted)}/{len(accepted)} hunk(s) applied — the rest were declined)"
        return output, False

    def _review_file_batch(
        self, calls: list[ToolCall]
    ) -> tuple[dict[str, tuple[str, bool]], dict[str, tuple[PendingChange, list[bool] | None]]]:
        """Review every write_file/edit_file call in `calls` as one
        continuous accept/reject stream (see UI.review_batch) instead of
        blocking on each file in turn.

        Returns (immediate, deferred): `immediate` call_ids already have
        their final (output, is_error) — invalid input never had anything to
        review. `deferred` call_ids have a decision but nothing written to
        disk yet; the caller commits each one at that call's original
        position in the round via `_finalize_pending_change`, so a
        non-file tool call interleaved in the same round still runs in the
        model's original order relative to these writes — only the
        interactive review itself is pulled together up front.
        """
        immediate: dict[str, tuple[str, bool]] = {}
        pending: list[PendingChange] = []
        for call in calls:
            try:
                pending.append(self._prepare_pending_change(call))
            except ToolError as exc:
                immediate[call.id] = (str(exc), True)
            except KeyError as exc:
                immediate[call.id] = (f"Missing required parameter: {exc}", True)

        deferred: dict[str, tuple[PendingChange, list[bool] | None]] = {
            p.call_id: (p, None) for p in pending if not (p.is_new or p.hunks)
        }
        reviewable = [p for p in pending if p.is_new or p.hunks]
        if reviewable:
            decisions = self.ui.review_batch([
                FileReview(call_id=p.call_id, path=p.path, is_new=p.is_new,
                           hunks=p.hunks, new_content=p.new_content)
                for p in reviewable
            ])
            deferred.update((p.call_id, (p, decisions[p.call_id])) for p in reviewable)
        return immediate, deferred

    def _execute_round(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute one round's tool calls in order. A round with more than
        one write_file/edit_file call has those calls reviewed together as a
        single batch (a "N files will change" summary, then every file's
        hunks in one continuous stream) instead of one file at a time; a
        round with zero or one such call keeps the original single-file
        review flow unchanged."""
        file_calls = [c for c in tool_calls if c.name in MUTATING_FILE_TOOLS]
        immediate: dict[str, tuple[str, bool]] = {}
        deferred: dict[str, tuple[PendingChange, list[bool] | None]] = {}
        if not self.config.yolo and len(file_calls) > 1:
            immediate, deferred = self._review_file_batch(file_calls)

        results = []
        for call in tool_calls:
            self.ui.tool_call(call.name, call.input)
            if call.id in immediate:
                output, is_error = immediate[call.id]
            elif call.id in deferred:
                output, is_error = self._finalize_pending_change(*deferred[call.id])
            else:
                output, is_error = self._execute_tool(call.name, call.input)
            self.ui.tool_result(output, is_error)
            results.append(ToolResult(call_id=call.id, output=output, is_error=is_error))
        return results

    def _commit(self, path: str, content: str) -> str:
        """Checkpoint the pre-change content, write `content`, and invalidate
        the code_search index — the one place write_file/edit_file actually
        touch disk, after approval has already been settled."""
        self._checkpoint_before(path)
        output = self.workspace.apply_content(path, content)
        self._code_index = None
        return output

    def _execute_mcp_tool(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """MCP tools are third-party code we don't control the effects of, so
        — unlike the built-in read-only tools — they're confirmed by default.
        The one exception is a server explicitly marking a tool `readOnlyHint`
        in its own annotations, which we trust the same way we trust our own
        read_file/search not to need confirmation."""
        tool = self.mcp.find_tool(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        if not tool.read_only and not self.config.yolo:
            if not self.ui.confirm(f"allow MCP tool {name} to run with {tool_input!r}?"):
                return DECLINED, True
        return self.mcp.call_tool(name, tool_input)

    def _get_code_index(self) -> CodeIndex:
        if self._code_index is None:
            self.reindex()
        return self._code_index

    def reindex(self) -> CodeIndex:
        """Force a fresh rebuild of the code_search index right now. Normally
        lazy — this is for /reindex, when the user wants an eager, on-demand
        build (e.g. right after pulling in a lot of new files some other
        way). Still cheap for files that haven't changed, thanks to the
        on-disk cache."""
        self._code_index = build_index(self.workspace, cache_dir=self._code_index_cache_dir)
        return self._code_index

    def _checkpoint_before(self, path: str) -> None:
        """Record `path`'s pre-turn content, if checkpointing is on and the
        path resolves inside the workspace. Never lets checkpointing itself
        block or fail the actual tool call."""
        if not self.config.checkpoints_enabled:
            return
        try:
            target = self.workspace.resolve(path)
        except ToolError:
            return
        try:
            self.checkpoints.record_before(target)
        except OSError:
            pass

    # -- context compaction ------------------------------------------------

    def token_budget(self) -> int:
        """Tokens available for context before the reserved output budget eats
        into the model's context window.

        The output reserve is capped at half the context window: config.max_tokens
        is a global default (64k) sized for large-context hosted models, but a
        small local model's context_window (an 8k default for anything on
        localhost) can't sensibly reserve more output than that — reserving the
        full 64k out of an 8k window would leave next to nothing for actual
        conversation and trigger compaction almost continuously.
        """
        reserve = min(self.config.max_tokens, self.backend.context_window // 2)
        return max(self.backend.context_window - reserve, MIN_TOKEN_BUDGET)

    def context_tokens_estimate(self) -> int:
        """Best known estimate of the current history's size in tokens: the
        real usage the backend reported after its last call, or a rough
        character-based heuristic if that isn't available yet."""
        if self.backend.last_context_tokens is not None:
            return self.backend.last_context_tokens
        return estimate_tokens_heuristic(self.messages, self.system_prompt())

    def maybe_compact(self, force: bool = False) -> bool:
        """Summarize older rounds into one message if the context is getting
        full (or always, when `force=True`, e.g. from /compact). Returns
        whether it actually compacted anything."""
        if not force:
            if self._compaction_unavailable:
                return False
            if self.context_tokens_estimate() < self.token_budget() * COMPACT_TRIGGER_RATIO:
                return False

        rounds = split_into_rounds(self.messages)
        keep = max(self.config.compact_keep_rounds, 1)
        if len(rounds) <= keep:
            return False  # nothing old enough to summarize yet

        to_summarize, to_keep = rounds[:-keep], rounds[-keep:]

        try:
            transcript = "\n\n".join(self.backend.render_round(r) for r in to_summarize)
            transcript = cap_head(transcript, MAX_SUMMARY_INPUT_CHARS)
            summary = self.backend.complete_text(SUMMARY_PROMPT, transcript)
        except Exception as exc:
            self._compaction_unavailable = True
            self.ui.error(f"context compaction unavailable ({exc}); continuing with full history")
            return False

        flat_keep = [m for r in to_keep for m in r]
        prefix = (
            f"[Summary of {len(to_summarize)} earlier turn(s), compacted to save context]\n"
            f"{summary.strip()}\n[end summary]\n\n"
        )
        if flat_keep and flat_keep[0].get("role") == "user" and isinstance(flat_keep[0].get("content"), str):
            flat_keep[0] = {**flat_keep[0], "content": prefix + flat_keep[0]["content"]}
        else:
            flat_keep.insert(0, {"role": "user", "content": prefix.strip()})

        before_count = len(self.messages)
        self.messages = flat_keep
        # Stale now that history changed shape; recomputed on the next real call.
        self.backend.last_context_tokens = None
        self.ui.info(
            f"context compacted: {before_count} -> {len(self.messages)} message(s) "
            f"({len(to_summarize)} earlier turn(s) summarized)"
        )
        return True

    # -- the loop ----------------------------------------------------------

    def run_turn(self, user_input: str, label: str | None = None) -> None:
        """Process one user message to completion (may involve many tool rounds).

        `label` is what gets recorded as this turn's checkpoint label (shown by
        /checkpoints); it defaults to `user_input` but callers that expand
        @mentions before calling run_turn should pass the raw, un-expanded text
        instead so the label stays a readable summary of what the user typed.
        """
        self.backend.add_user_message(self.messages, user_input)
        self.ui.assistant_prefix()
        self.checkpoints.begin_turn(label if label is not None else user_input)
        self._compaction_unavailable = False

        try:
            while True:
                self.maybe_compact()

                result = self.backend.stream_turn(
                    self.messages, self.system_prompt(), self.tool_definitions(), self.ui
                )

                if result.stop == "refusal":
                    self.ui.error("The request was declined by the model's safety system. Try rephrasing.")
                    return

                if result.stop == "length":
                    self.ui.error("Response hit the output token limit; it may be incomplete.")

                if not result.tool_calls:
                    if result.usage:
                        self.ui.info(f"tokens: {result.usage}")
                    return

                results = self._execute_round(result.tool_calls)
                self.backend.add_tool_results(self.messages, results)
        finally:
            # A disk error here (e.g. ~/.tythancode unwritable) must never mask
            # a real exception already propagating from the try block above —
            # a bare `raise` from a finally clause replaces it, which would
            # turn e.g. a network error into a confusing OSError instead.
            try:
                checkpoint = self.checkpoints.commit_turn()
            except OSError as exc:
                checkpoint = None
                self.ui.error(f"couldn't save checkpoint ({exc}); this turn's edits won't be undoable")
            if checkpoint:
                skipped = checkpoint.skipped_large + checkpoint.skipped_binary
                note = f" ({len(skipped)} large/binary file(s) not covered)" if skipped else ""
                if checkpoint.changes:
                    self.ui.info(
                        f"checkpoint saved: {len(checkpoint.changes)} file(s) changed{note} — /undo to revert"
                    )
                else:
                    self.ui.info(f"note: {len(skipped)} large/binary file(s) changed but aren't covered by /undo")
