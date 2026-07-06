# Tythan Code

**The AI coding agent that won't ship vulnerabilities.**

A Cursor-style AI coding assistant that lives in your terminal — with **any
model you want** and a **built-in security auditor**. Chat about your project;
the assistant reads and edits files, searches the codebase, runs shell
commands, and scans its own output for security issues before calling a task
done. Every file change is shown as a diff and every command waits for your
confirmation.

```
 _        _   _                           _
| |_ _  _| |_| |_  __ _ _ _    __ ___  __| |___
|  _| || |  _| ' \/ _` | ' \  / _/ _ \/ _` / -_)
 \__|\_, |\__|_||_\__,_|_||_| \__\___/\__,_\___|
     |__/
```

![`/audit` finding a leaked Stripe key, a SQL-injection-prone query and an eval() call in a sample file](docs/audit-demo.gif)

## Features

- **Security-first agent** — the `security_scan` tool detects leaked secrets
  and API keys (AWS, GitHub, Stripe, Google, Slack, Telegram, JWTs, Bearer
  tokens + a Shannon-entropy detector for everything else), dangerous code
  patterns (eval, pickle, SQL built from f-strings, `shell=True`,
  `verify=False`, weak ciphers, `random` used for secrets, ...) and insecure
  config (wildcard CORS, JWT `none`, debug mode, plain-http endpoints). The
  agent audits code it just wrote and fixes CRITICAL/HIGH findings before
  declaring a task done; run `/audit` any time for an instant offline report.
- **Dependency CVE check (SCA)** — pinned dependencies from
  `requirements.txt`, `pyproject.toml` and `package.json` are checked against
  the [OSV.dev](https://osv.dev) vulnerability database (free, no key).
  Included in `/audit`; the agent can request it via
  `security_scan(include_dependencies=true)`. Degrades gracefully offline.
- **Any model** — native Anthropic API plus *any* OpenAI-compatible endpoint:
  OpenAI, OpenRouter, Groq, DeepSeek, Mistral, xAI, and fully local models via
  Ollama / LM Studio / vLLM. Switch providers mid-session with `/provider`.
- **Agentic loop** — the model reads files, edits them, runs tests and
  iterates until the task is done, streaming its answer live.
- **Human in the loop, per hunk** — a new file gets a single whole-file diff
  and `[y/N]`; a change to an existing file is split into independent hunks
  (`git add -p` style) so you can accept some and reject others in one
  response instead of an all-or-nothing yes/no — `y`/`n` per hunk, `a` to
  accept it and everything remaining, `d` to reject it and everything
  remaining. Only the accepted hunks are written; the tool result tells the
  model how many of the total were actually applied. Shell commands still
  ask a plain yes/no before running. `--yolo` turns all of this off.
- **Batched review across a whole round** — when the model changes several
  files in the same response (e.g. renaming something and updating every
  call site), you're not blocked on each file one at a time: a summary shows
  how many files will change, then every file's hunks play out as one
  continuous `y/n/a/d` stream — `a`/`d` accept or reject everything
  remaining across the *entire* batch, not just the current file. The system
  prompt also nudges the model to actually send those files together in one
  response instead of trickling them out one write at a time. A shell
  command in the same round still runs at its original point relative to
  the writes around it — only the review itself is pulled together.
- **`@` mentions** — `@src/app.py` attaches that file's contents;
  `@src/utils/` (a directory) lists everything inside it (recursively,
  `.gitignore`-aware) so the model can `read_file` whichever ones actually
  matter instead of you attaching them one by one; `@git:diff`, `@git:staged`
  and `@git:status` attach the workspace's current git state — handy for
  "review what I've changed" or "finish this in-progress commit" without a
  confirmable `run_command` round-trip, since these run a fixed, hardcoded
  git command with no user input in the argv.
- **Workspace-confined** — all file operations are locked inside the project
  directory; path traversal is rejected.
- **`.gitignore`-aware** — `list_files`, `search`, `code_search`, `/audit` and
  the dependency check all skip whatever the project's own `.gitignore`
  excludes, on top of the usual junk directories (`.git`, `node_modules`,
  ...). No more noise from build output or vendored code that happens to live
  outside those hardcoded names.
- **Project rules** — drop a `.tythancode/rules.md`, `.tythancoderules`,
  `AGENTS.md` or `.cursorrules` file (checked in that order, first one found
  wins) in the workspace root and Tythan Code loads it into every system
  prompt automatically: conventions, build/test commands, things it should
  never touch. No more repeating the same context every session; the banner
  tells you when one was picked up.
- **Task plan tracking** — for multi-step work the assistant maintains a
  visible todo list (`todo_write`), shown as a checklist as it works through
  each step. Check it any time with `/todos`.
- **Ranked codebase search (`code_search`)** — BM25 keyword ranking over
  chunked files, for when you don't know the exact string to grep for ("where
  is retry logic for HTTP calls?"). This is lexical ranking, not embeddings —
  fully offline, no model call, no vector database — but it recovers a lot of
  what naive grep misses, including matching `read_file`/`readFile` from a
  query like "read file". Built lazily on first use, then cached on disk
  under `~/.tythancode/index/` keyed by each file's modification time and
  size, so re-opening the same project later only re-tokenizes what actually
  changed instead of the whole codebase again. `/reindex` forces an eager
  rebuild (still fast, for the same reason).
- **MCP client** — connect any [Model Context Protocol](https://modelcontextprotocol.io)
  stdio server (filesystem, fetch, databases, your own internal tools, ...)
  by adding it under `"mcp_servers"` in `~/.tythancode/config.json`. Their
  tools show up to the model as `mcp__<server>__<tool>` alongside the
  built-ins. Tythan Code doesn't control what a third-party server's tool
  actually does, so every call is confirmed before running — unless the
  server itself marks the tool `readOnlyHint`, the same way `read_file`
  needs no confirmation. `/mcp` shows what's connected; a server that fails
  to start is reported and simply contributes no tools rather than blocking
  startup.
- **Web fetch (`fetch_url`)** — read documentation, changelogs or an API
  response straight into the conversation, the terminal equivalent of
  Cursor's `@web`. Confirmed before running, same as `run_command`, and
  every request — plus every individual redirect hop — is resolved and
  checked against private/loopback/link-local/reserved address ranges
  before connecting, so a prompt-injected "fetch http://169.254.169.254/..."
  can't be used to reach cloud metadata endpoints or internal services.
- **Tools** — `read_file`, `write_file`, `edit_file` (exact string replace),
  `list_files` (glob), `search` (regex), `code_search` (ranked), `todo_write`,
  `fetch_url`, `run_command`, plus whatever connected MCP servers add.
- **Background agent (`--branch`)** — run a task unattended (`-p "..." --branch --yolo`)
  on a fresh, auto-named git branch instead of whatever's currently checked
  out, so it can't collide with work in progress; leftover uncommitted
  changes are committed automatically when the run ends. Give it an explicit
  name (`--branch fix-42`) to land on a specific branch instead. Not a git
  repo, or branch creation fails? Tythan Code says so and just runs on the
  current branch instead of refusing the task outright.
- **Undo (`/undo`)** — every `write_file`/`edit_file` is checkpointed before it
  runs. `/undo` reverts the whole last turn's file changes in one step,
  survives restarting Tythan Code, and stays out of the way otherwise: nothing
  is written to `~/.tythancode` until a file actually changes. Shell commands
  run via `run_command` aren't covered — there's no honest way to snapshot and
  revert arbitrary shell effects, so this is a safety net for agent-authored
  edits, not a full undo of everything the agent does. Files over 5MB or that
  aren't valid UTF-8 are skipped rather than checkpointed (so `/undo` never
  "restores" a lossy, corrupted copy) — Tythan Code tells you when this
  happens. `/checkpoints [n]` lists recent checkpoints (10 by default, up to
  the 50 retained per workspace).
- **Automatic context compaction** — long sessions don't hit a hard
  context-length error. When the conversation approaches the model's context
  window, Tythan Code summarizes the older turns into one message and keeps
  the most recent turns verbatim, so the assistant keeps working instead of
  failing outright. Trigger it manually with `/compact`, check usage with
  `/context`.

## Install

```bash
pip install tythan-code
```

Or from source:

```bash
git clone https://github.com/TythanAI/Tythan-Code.git
cd Tythan-Code
pip install .
```

Requires Python 3.10+. For the default Anthropic provider:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or: ant auth login   (the SDK picks the profile up automatically)
```

## Providers

On first run Tythan Code creates `~/.tythancode/config.json`:

```json
{
  "default_provider": "anthropic",
  "providers": {
    "anthropic":  {"type": "anthropic", "model": "claude-opus-4-8"},
    "openai":     {"type": "openai", "base_url": "https://api.openai.com/v1",
                   "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
    "openrouter": {"type": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "api_key_env": "OPENROUTER_API_KEY", "model": "anthropic/claude-sonnet-4.5"},
    "ollama":     {"type": "openai", "base_url": "http://localhost:11434/v1",
                   "model": "qwen2.5-coder:14b"}
  }
}
```

Add any OpenAI-compatible service as a new entry (`type: "openai"` +
`base_url` + `api_key_env` + `model`). Local endpoints (localhost) don't need
a key. `type: "anthropic"` uses the native Anthropic API with adaptive
thinking, effort control and prompt caching.

Each provider entry can also set `"context_window": <tokens>` to override how
much context Tythan Code assumes that model has before it proactively
compacts history. Without it, Tythan Code guesses conservatively: 200k for
Anthropic, 128k for known hosted APIs (OpenAI, OpenRouter, Groq, ...), and a
cautious 8k for anything on localhost — local model servers commonly run with
a much smaller context than the underlying model supports unless configured
otherwise, so set this explicitly if you've raised `num_ctx` (Ollama) or
similar.

## MCP servers

Add stdio MCP servers under `"mcp_servers"` in the same config file:

```json
{
  "mcp_servers": {
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]}
  }
}
```

`command`/`args` launch the server as a subprocess over stdio (the same
transport `npx`/`uvx`-based servers use); `env` (optional) adds extra
environment variables on top of a safe inherited default (`PATH`, `HOME`,
...) — it does not replace the environment outright. Tythan Code connects to
every configured server at startup; a server that fails to start is reported
once and simply has no tools, it doesn't stop Tythan Code from working.

## Usage

```bash
tythancode ~/my-project              # interactive chat in that workspace
tythancode                           # current directory
tythancode -p "fix the failing test" # one-shot, non-interactive
tythancode --provider ollama         # pick a provider for this session
tythancode --model claude-sonnet-5 --effort xhigh
tythancode --yolo                    # auto-approve everything (careful!)
tythancode --no-checkpoints          # don't record file checkpoints (disables /undo)
tythancode -p "add retries to the HTTP client" --branch --yolo  # background agent: isolated branch, no prompts
tythancode -p "fix issue #42" --branch fix-42                   # same, on a specific branch name
```

### In-chat commands

| Command | Effect |
|---|---|
| `/help` | show help |
| `/clear` | reset the conversation |
| `/provider [name]` | list providers / switch (resets the chat) |
| `/model <id>` | switch model within the current provider (context window stays as configured for the provider — see `/context`) |
| `/effort <lvl>` | `low` / `medium` / `high` / `xhigh` / `max` (Anthropic) |
| `/audit [path]` | offline security scan of the workspace (or a subpath) |
| `/yolo` | toggle confirmation prompts |
| `/undo` | revert the file changes from the last turn |
| `/checkpoints [n]` | list recent undo checkpoints (10 by default) |
| `/todos` | show the assistant's current task plan |
| `/reindex` | rebuild the `code_search` index now (otherwise built lazily on first use) |
| `/mcp` | show connected MCP servers and the tools they expose |
| `/compact` | summarize older history now to free up context |
| `/context` | show estimated context usage vs. the model's window |
| `/exit` or Ctrl+D | quit |

### Example session

```
you> add a --verbose flag to @cli.py
assistant
⚙ read_file {"path": "cli.py"}
⚙ edit_file {"path": "cli.py", "old_string": "...", "new_string": "..."}
╭─ cli.py — hunk 1/1 ───────────────────────────╮
│ @@ -12,1 +12,1 @@                             │
│ -    parser.add_argument("--quiet", ...)      │
│ +    parser.add_argument("--verbose", ...)    │
╰────────────────────────────────────────────────╯
apply this hunk? [y/n/a/d/?] y
Done — added the flag and wired it to the logger setup.
tokens: 8231 in / 412 out / 7100 cached
```

## Development

```bash
pip install -e ".[dev]"
pytest           # tools, agent loop and provider tests run offline
```

## Architecture

```
tythancode/
├── cli.py                    # REPL, slash commands, @mentions
├── agent.py                  # provider-agnostic agent loop + confirmations
├── tools.py                  # tool schemas + sandboxed executors
├── ui.py                     # rich rendering: streams, diffs, prompts
├── security.py               # offline security scanner (/audit + agent tool)
├── sca.py                    # dependency CVE check via OSV.dev
├── config.py                 # ~/.tythancode/config.json provider registry
├── compaction.py             # round-splitting + token-estimate helpers for /compact
├── checkpoints.py            # file-level undo store behind /undo, /checkpoints
├── storage.py                # shared ~/.tythancode per-workspace directory hashing
├── ignore.py                 # .gitignore matching shared by tools/security/sca
├── rules.py                  # project rules file discovery (AGENTS.md, ...)
├── codeindex.py              # BM25 index + ranking behind the code_search tool
├── hunks.py                  # per-hunk diff split/reconstruct behind write_file/edit_file review
├── mcp_client.py             # MCP (Model Context Protocol) stdio client
├── webfetch.py               # fetch_url tool: SSRF-checked HTTP fetch + HTML->text
├── background.py             # --branch: git-branch isolation for unattended runs
└── providers/
    ├── base.py               # Backend interface (owns native msg format)
    ├── anthropic_backend.py  # Messages API: streaming, thinking, caching
    └── openai_backend.py     # any /v1/chat/completions endpoint
```

The agent loop is provider-agnostic: one streaming call per round; when the
model returns tool calls, Tythan Code executes them locally (asking you first
for anything mutating), sends results back and repeats until the turn ends.
Refusals, `pause_turn` and token-limit stops are handled explicitly.

Two pieces of session-level bookkeeping wrap that loop:

- **Compaction** (`compaction.py` + `Agent.maybe_compact`) splits the message
  history into "rounds" (a user turn plus everything the agent did in
  response), and — once the estimated context in use crosses ~80% of the
  model's context window minus the reserved output budget — asks the backend
  to summarize every round except the most recent `compact_keep_rounds` into
  one message. Each `Backend` implements `render_round` (native messages →
  plain text) and `complete_text` (one-shot, tool-free completion) to make
  this provider-agnostic; if the summarization call itself fails, Tythan Code
  logs it once and keeps working with the full history rather than looping on
  a broken call.
- **Checkpoints** (`checkpoints.py` + `Agent._checkpoint_before`) record each
  touched file's pre-turn content the first time `write_file`/`edit_file`
  touches it in a turn, and persist the whole turn as one checkpoint under
  `~/.tythancode/checkpoints/<hash of the workspace path>/`. `/undo` pops the
  most recent one and restores every file it touched.
- **Code index** (`codeindex.py` + `Agent._get_code_index`) is built lazily on
  the first `code_search` call and kept in memory for the rest of the
  session. `Agent` drops the in-memory instance after `write_file`,
  `edit_file` or `run_command` (any tool that could have changed files on
  disk), so the next search always reflects current content instead of
  silently going stale. Rebuilding is also backed by a per-workspace JSON
  cache under `~/.tythancode/index/<hash of the workspace path>/index.json`
  keyed by each file's `(mtime, size)`: a file that hasn't changed reuses its
  cached chunks instead of being re-read and re-tokenized. The cache format
  is versioned (`codeindex.CACHE_VERSION`) so a future change to chunking or
  tokenization invalidates old caches automatically, and any read/parse
  failure just falls back to a full rebuild rather than serving corrupt data.
