"""Built-in security scanner — the thing other terminal agents don't have.

Fast, offline, regex-based checks for the highest-impact classes of issues:
leaked secrets, dangerous code patterns, and insecure configuration. Exposed
both as a model tool (security_scan) and a direct /audit command, so the
agent can audit the code it just wrote — or fix what the scan finds.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .tools import Workspace

MAX_FILE_BYTES = 1_000_000
MAX_FINDINGS = 200

# (rule_id, severity, message, compiled regex)
# Severities: CRITICAL > HIGH > MEDIUM
RULES: list[tuple[str, str, str, re.Pattern]] = [
    # --- secrets ---------------------------------------------------------
    ("SEC-AWS-KEY", "CRITICAL", "AWS access key ID in source",
     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SEC-PRIVATE-KEY", "CRITICAL", "Private key material in source",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----")),
    ("SEC-GITHUB-TOKEN", "CRITICAL", "GitHub token in source",
     re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("SEC-SLACK-TOKEN", "CRITICAL", "Slack token in source",
     re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("SEC-API-PROVIDER-KEY", "CRITICAL", "AI provider API key in source",
     re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("SEC-STRIPE-KEY", "CRITICAL", "Stripe live secret key in source",
     re.compile(r"\bsk_live_[0-9a-zA-Z]{24,}\b")),
    ("SEC-GOOGLE-KEY", "CRITICAL", "Google API key in source",
     re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("SEC-TELEGRAM-TOKEN", "CRITICAL", "Telegram bot token in source",
     re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}\b")),
    ("SEC-JWT-HARDCODED", "MEDIUM", "Hardcoded JWT in source",
     re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}")),
    ("SEC-BEARER-TOKEN", "HIGH", "Hardcoded Bearer token",
     re.compile(r"""(?i)authorization['"]?\s*[:=]\s*["']bearer\s+[A-Za-z0-9._~+/-]{20,}""")),
    ("SEC-HARDCODED-CRED", "HIGH", "Hardcoded credential assignment",
     re.compile(r"""(?i)\b(?:password|passwd|secret|api_key|apikey|auth_token|access_token)\s*[:=]\s*["'][^"'\s]{8,}["']""")),
    # --- dangerous code (Python) -----------------------------------------
    ("PY-EVAL", "HIGH", "eval()/exec() — arbitrary code execution risk",
     re.compile(r"\b(?:eval|exec)\s*\(")),
    ("PY-PICKLE", "HIGH", "pickle.loads on untrusted data is code execution",
     re.compile(r"\bpickle\.loads?\s*\(")),
    ("PY-YAML-LOAD", "MEDIUM", "yaml.load without SafeLoader",
     re.compile(r"\byaml\.load\s*\((?![^)]*SafeLoader)")),
    ("PY-SHELL-TRUE", "MEDIUM", "subprocess with shell=True — injection risk if input is dynamic",
     re.compile(r"\bshell\s*=\s*True\b")),
    ("PY-OS-SYSTEM", "MEDIUM", "os.system — prefer subprocess without a shell",
     re.compile(r"\bos\.system\s*\(")),
    ("PY-SQL-FSTRING", "HIGH", "SQL built with f-string — use parameterized queries",
     re.compile(r"""\b(?:execute|executemany)\s*\(\s*f["']""")),
    ("PY-VERIFY-FALSE", "HIGH", "TLS verification disabled (verify=False)",
     re.compile(r"\bverify\s*=\s*False\b")),
    ("PY-MD5-PASSWORD", "MEDIUM", "MD5/SHA1 are not password hashes — use bcrypt/argon2",
     re.compile(r"\bhashlib\.(?:md5|sha1)\s*\([^)]*passw", re.IGNORECASE)),
    ("PY-FLASK-DEBUG", "MEDIUM", "Flask debug mode in code — remote code execution if deployed",
     re.compile(r"\.run\s*\([^)]*debug\s*=\s*True")),
    ("PY-DJANGO-DEBUG", "MEDIUM", "DEBUG = True — never deploy with debug enabled",
     re.compile(r"^\s*DEBUG\s*=\s*True\b")),
    ("PY-WEAK-CIPHER", "HIGH", "Weak cipher/mode (DES or ECB)",
     re.compile(r"\bMODE_ECB\b|\bDES\.new\s*\(|algorithms\.(?:TripleDES|Blowfish|ARC4)\b")),
    ("PY-INSECURE-RANDOM", "HIGH", "random module used for a secret — use the secrets module",
     re.compile(r"(?i)\b(?:token|secret|otp|nonce|session_key)\w*\s*=\s*.*\brandom\.(?:random|randint|choice|randrange|getrandbits)\b")),
    ("PY-MKTEMP", "MEDIUM", "tempfile.mktemp is race-prone — use mkstemp/NamedTemporaryFile",
     re.compile(r"\btempfile\.mktemp\s*\(")),
    ("PY-REQUEST-OPEN", "MEDIUM", "File path taken from request input — path traversal risk",
     re.compile(r"\b(?:open|send_file)\s*\([^)]*request\.")),
    ("GO-SHELL-EXEC", "MEDIUM", "Shell exec with -c — injection risk if input is dynamic",
     re.compile(r"""exec\.Command\(\s*["'](?:sh|bash)["']\s*,\s*["']-c["']""")),
    ("PHP-DANGEROUS-EXEC", "HIGH", "Dynamic code/command execution on a variable",
     re.compile(r"\b(?:eval|system|shell_exec|passthru|popen)\s*\(\s*\$")),
    ("NET-PLAIN-HTTP", "MEDIUM", "Unencrypted http:// endpoint — use https",
     re.compile(r"""["']http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\{)[^"'\s]+["']""")),
    # --- dangerous code (JS/TS) ------------------------------------------
    ("JS-EVAL", "HIGH", "eval()/new Function() — arbitrary code execution risk",
     re.compile(r"\bnew\s+Function\s*\(")),
    ("JS-CHILD-EXEC", "MEDIUM", "child_process.exec with dynamic input — injection risk",
     re.compile(r"\bchild_process\b.*\bexec\s*\(|\brequire\(['\"]child_process['\"]\)")),
    ("JS-INNERHTML", "MEDIUM", "innerHTML/dangerouslySetInnerHTML — XSS risk",
     re.compile(r"\.innerHTML\s*=|dangerouslySetInnerHTML")),
    # --- insecure config ---------------------------------------------------
    ("CFG-CORS-WILDCARD", "MEDIUM", "CORS allows any origin",
     re.compile(r"""Access-Control-Allow-Origin['"]?\s*[:=,]\s*['"]\*""")),
    ("CFG-JWT-NONE", "HIGH", "JWT signature verification weakened",
     re.compile(r"""algorithms?\s*[:=]\s*\[?\s*['"]none['"]|verify_signature['"]?\s*:\s*False""")),
    ("CFG-BIND-ALL-DEBUG", "MEDIUM", "Service bound to 0.0.0.0 — make sure that's intended",
     re.compile(r"""['"]0\.0\.0\.0['"]""")),
]

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}

# File types worth scanning.
SCAN_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rb", ".php",
    ".java", ".kt", ".rs", ".c", ".cc", ".cpp", ".h", ".cs", ".swift", ".sh",
    ".bash", ".zsh", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".tf", ".sql", ".html", ".vue", ".svelte",
}

# Lines that look like test fixtures / examples get downgraded, not dropped.
FIXTURE_HINT = re.compile(r"(?i)example|sample|dummy|fake|test|xxx+|placeholder|your[_-]?key")

# Generated/vendored files that produce only noise.
SKIP_FILENAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "composer.lock"}
SKIP_NAME_PARTS = (".min.", ".map", ".bundle.")

# --- entropy-based generic secret detection -------------------------------

ENTROPY_CANDIDATE = re.compile(r"""["']([A-Za-z0-9+/=_\-]{24,})["']""")
ENTROPY_CONTEXT = re.compile(r"(?i)\b\w*(?:key|token|secret|passw|credential|auth)\w*\s*[:=]")
ENTROPY_THRESHOLD = 4.4  # bits/char; random base64 ≈ 6, English text ≈ 3–4


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def entropy_findings(line: str, lineno: int, rel_path: str) -> list[Finding]:
    """Flag high-entropy strings assigned to secret-looking names."""
    if not ENTROPY_CONTEXT.search(line):
        return []
    out = []
    for match in ENTROPY_CANDIDATE.finditer(line):
        value = match.group(1)
        if shannon_entropy(value) < ENTROPY_THRESHOLD:
            continue
        sev = "MEDIUM" if FIXTURE_HINT.search(line) else "HIGH"
        out.append(Finding(
            rule="SEC-HIGH-ENTROPY",
            severity=sev,
            message="High-entropy string assigned to a secret-looking name",
            path=rel_path,
            line=lineno,
            snippet=line.strip()[:160],
        ))
    return out


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    path: str
    line: int
    snippet: str


def scan_text(text: str, rel_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000:
            continue  # minified bundles
        matched_rules = set()
        for rule_id, severity, message, rx in RULES:
            if rx.search(line):
                matched_rules.add(rule_id)
                sev = severity
                if severity == "CRITICAL" and FIXTURE_HINT.search(line):
                    sev = "MEDIUM"
                findings.append(Finding(
                    rule=rule_id,
                    severity=sev,
                    message=message,
                    path=rel_path,
                    line=lineno,
                    snippet=line.strip()[:160],
                ))
        # Entropy check only when no specific secret rule already fired.
        if not matched_rules & {"SEC-AWS-KEY", "SEC-GITHUB-TOKEN", "SEC-SLACK-TOKEN",
                                "SEC-API-PROVIDER-KEY", "SEC-STRIPE-KEY", "SEC-GOOGLE-KEY",
                                "SEC-TELEGRAM-TOKEN", "SEC-HARDCODED-CRED", "SEC-BEARER-TOKEN"}:
            findings.extend(entropy_findings(line, lineno, rel_path))
    return findings


def scan_workspace(ws: Workspace, subpath: str = ".") -> list[Finding]:
    root = ws.resolve(subpath)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    findings: list[Finding] = []
    for p in files:
        if not p.is_file():
            continue
        rel = p.relative_to(ws.root)
        if ws.is_ignored(rel):
            continue
        if p.name in SKIP_FILENAMES or any(part in p.name for part in SKIP_NAME_PARTS):
            continue
        if p.suffix.lower() not in SCAN_SUFFIXES and p.name != ".env":
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        findings.extend(scan_text(text, str(rel)))
        if len(findings) >= MAX_FINDINGS:
            break
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.path, f.line))
    return findings[:MAX_FINDINGS]


def format_findings(findings: list[Finding]) -> str:
    """Plain-text report for the model / logs."""
    if not findings:
        return "No security findings. Scanned for secrets, dangerous code patterns and insecure config."
    lines = [f"{len(findings)} security finding(s):", ""]
    for f in findings:
        lines.append(f"[{f.severity}] {f.rule} {f.path}:{f.line} — {f.message}")
        lines.append(f"    {f.snippet}")
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: SEVERITY_ORDER[kv[0]]))
    lines += ["", f"Summary: {summary}"]
    return "\n".join(lines)
