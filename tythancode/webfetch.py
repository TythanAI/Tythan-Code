"""fetch_url tool: pull external web content (docs, API responses, release
notes, ...) into the conversation — the closest terminal-agent equivalent of
Cursor's "@web".

Network egress triggered by an LLM tool call is a real SSRF vector if
anything the model reads (a file, a tool result, a prompt-injected comment)
can influence the URL: "fetch http://169.254.169.254/latest/meta-data/" is
the canonical cloud-metadata-theft payload, and "fetch http://localhost:9200/"
would reach whatever's listening on the user's own machine. So, on top of
the same y/N confirmation `run_command` gets, every URL — and every redirect
hop, individually — is resolved and checked against private/loopback/
link-local/reserved ranges before it's ever connected to. This is a
pre-connect check, not a pinned-connection one: a DNS answer could in theory
change between the check and the actual connect (DNS rebinding). That's a
known, accepted gap here — closing it fully means bypassing the HTTP
client's own connection handling to connect to a pinned IP, which is real
extra complexity for a threat model (an attacker who controls both DNS
timing and response content) well beyond what prompt-injected page content
can achieve on its own.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from urllib.parse import urlparse

FETCH_TIMEOUT = 15.0
MAX_FETCH_BYTES = 2_000_000
MAX_RETURNED_CHARS = 20_000
MAX_REDIRECTS = 5
USER_AGENT = "tythancode-fetch/1.0"


class FetchError(Exception):
    """Raised for any fetch failure; reported back to the model as is_error."""


def _resolve_all(hostname: str) -> list[str]:
    infos = socket.getaddrinfo(hostname, None)
    return sorted({info[4][0] for info in infos})


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(url: str, resolver=_resolve_all) -> str:
    """Raise FetchError unless `url` is an http(s) URL whose host resolves
    only to public addresses. Returns `url` unchanged on success."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"Only http:// and https:// URLs are supported, got: {parsed.scheme or url!r}")
    hostname = parsed.hostname
    if not hostname:
        raise FetchError(f"URL has no hostname: {url!r}")
    try:
        addrs = resolver(hostname)
    except socket.gaierror as exc:
        raise FetchError(f"Cannot resolve host {hostname!r}: {exc}") from exc
    if not addrs:
        raise FetchError(f"Host {hostname!r} did not resolve to any address")
    if not all(_is_public_ip(a) for a in addrs):
        raise FetchError(
            f"Refusing to fetch a private/internal/loopback address "
            f"({hostname!r} resolves to {', '.join(addrs)})"
        )
    return url


_SCRIPT_STYLE_RX = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RX = re.compile(r"<[^>]+>")
_INLINE_WS_RX = re.compile(r"[ \t]+")
_BLANK_LINES_RX = re.compile(r"\n{3,}")


def html_to_text(raw_html: str) -> str:
    """Small, dependency-free HTML -> text: drop script/style blocks and all
    other tags, unescape entities, collapse whitespace. Not a real renderer
    (no layout, no link footnotes) — good enough for an LLM to read
    documentation prose, not for anything that needs the page's structure."""
    text = _SCRIPT_STYLE_RX.sub(" ", raw_html)
    text = _TAG_RX.sub(" ", text)
    text = html.unescape(text)
    text = _INLINE_WS_RX.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RX.sub("\n\n", text)


def _default_get(url: str):
    import httpx

    return httpx.get(
        url, timeout=FETCH_TIMEOUT, follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )


def fetch_url(url: str, get=None, resolver=_resolve_all) -> str:
    """Fetch `url` and return readable text, truncated to a safe size.

    Redirects are followed manually (not by the HTTP client) so each hop's
    target gets the same SSRF check as the original URL before it's
    connected to. `get`/`resolver` are overridable for tests.
    """
    get = get or _default_get
    current = url
    resp = None
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_url(current, resolver=resolver)
        try:
            resp = get(current)
        except Exception as exc:
            raise FetchError(f"Request to {current} failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            if not location:
                raise FetchError(f"HTTP {resp.status_code} redirect with no Location header")
            current = location
            continue
        break
    else:
        raise FetchError(f"Too many redirects (>{MAX_REDIRECTS})")

    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} fetching {current}")

    content = resp.content[:MAX_FETCH_BYTES]
    encoding = getattr(resp, "encoding", None) or "utf-8"
    try:
        text_body = content.decode(encoding, errors="replace")
    except LookupError:
        text_body = content.decode("utf-8", errors="replace")

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type.lower():
        text_body = html_to_text(text_body)

    if len(text_body) > MAX_RETURNED_CHARS:
        text_body = text_body[:MAX_RETURNED_CHARS] + "\n... [truncated, the page was longer]"
    return text_body or "(empty response)"
