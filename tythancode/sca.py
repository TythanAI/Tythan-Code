"""Dependency vulnerability check (SCA) via the free OSV.dev database.

Parses Python and JS manifests for pinned versions and queries OSV's batch
API. Network-dependent by nature: failures degrade to a notice, never crash
the audit. No API key required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .ignore import GitignoreMatcher
from .security import Finding
from .tools import SKIP_DIRS

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_TIMEOUT = 15.0
MAX_DEPS = 300

REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([0-9][^\s;#]*)")
PEP508_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([0-9][^\s;#,\"']*)")
NPM_EXACT = re.compile(r"^[~^]?v?(\d+\.\d+\.\d+(?:[-.][\w.]+)?)$")


@dataclass
class Dependency:
    ecosystem: str  # "PyPI" | "npm"
    name: str
    version: str
    manifest: str  # which file it came from


def parse_requirements(text: str, manifest: str) -> list[Dependency]:
    deps = []
    for line in text.splitlines():
        m = REQ_LINE.match(line)
        if m:
            deps.append(Dependency("PyPI", m.group(1).lower(), m.group(2), manifest))
    return deps


def parse_pyproject(text: str, manifest: str) -> list[Dependency]:
    """Extract pinned deps from [project] dependencies without a TOML parser dep."""
    deps = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies") and "[" in stripped:
            in_deps = True
        elif in_deps and stripped.startswith("]"):
            in_deps = False
        if in_deps:
            for spec in re.findall(r"""["']([^"']+)["']""", stripped):
                m = PEP508_PIN.match(spec)
                if m:
                    deps.append(Dependency("PyPI", m.group(1).lower(), m.group(2), manifest))
    return deps


def parse_package_json(text: str, manifest: str) -> list[Dependency]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    deps = []
    for section in ("dependencies", "devDependencies"):
        for name, spec in (data.get(section) or {}).items():
            m = NPM_EXACT.match(str(spec))
            if m:
                deps.append(Dependency("npm", name, m.group(1), manifest))
    return deps


def collect_dependencies(root: Path) -> list[Dependency]:
    ignore = GitignoreMatcher.load(root)
    deps: list[Dependency] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        if ignore.matches(rel.as_posix(), is_dir=False):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.name == "requirements.txt" or (p.name.startswith("requirements") and p.suffix == ".txt"):
            deps.extend(parse_requirements(text, str(rel)))
        elif p.name == "pyproject.toml":
            deps.extend(parse_pyproject(text, str(rel)))
        elif p.name == "package.json":
            deps.extend(parse_package_json(text, str(rel)))
        if len(deps) >= MAX_DEPS:
            break
    # de-duplicate
    seen = set()
    unique = []
    for d in deps:
        key = (d.ecosystem, d.name, d.version)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique[:MAX_DEPS]


def _default_post(url: str, payload: dict) -> dict:
    import httpx

    resp = httpx.post(url, json=payload, timeout=OSV_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def query_osv(deps: list[Dependency], post=None) -> list[Finding]:
    """Query OSV for each dependency; returns findings for vulnerable ones."""
    if not deps:
        return []
    post = post or _default_post
    payload = {
        "queries": [
            {"package": {"name": d.name, "ecosystem": d.ecosystem}, "version": d.version}
            for d in deps
        ]
    }
    data = post(OSV_BATCH_URL, payload)
    findings = []
    for dep, result in zip(deps, data.get("results", [])):
        vulns = result.get("vulns") or []
        if not vulns:
            continue
        ids = [v.get("id", "?") for v in vulns]
        shown = ", ".join(ids[:5]) + ("…" if len(ids) > 5 else "")
        findings.append(Finding(
            rule=f"SCA-{dep.ecosystem.upper()}",
            severity="HIGH",
            message=f"{dep.name} {dep.version} has {len(ids)} known vulnerabilit{'y' if len(ids) == 1 else 'ies'} ({shown}) — upgrade",
            path=dep.manifest,
            line=0,
            snippet=f"{dep.name}=={dep.version}",
        ))
    return findings


def scan_dependencies(root: Path, post=None) -> tuple[list[Finding], str]:
    """Full SCA pass. Returns (findings, status_note)."""
    deps = collect_dependencies(root)
    if not deps:
        return [], "no pinned dependencies found (only exact versions are checked)"
    try:
        findings = query_osv(deps, post=post)
    except Exception as exc:
        return [], f"dependency check skipped — OSV.dev unreachable ({type(exc).__name__})"
    note = f"checked {len(deps)} pinned dependencies against OSV.dev"
    return findings, note
