import socket

import pytest

from tythancode.webfetch import FetchError, fetch_url, html_to_text, validate_public_url


def resolver_for(mapping):
    """Fake resolver: hostname -> list of IPs, for tests (no real DNS)."""
    def resolve(hostname):
        if hostname not in mapping:
            raise socket.gaierror(f"no fake entry for {hostname}")
        return mapping[hostname]
    return resolve


PUBLIC = resolver_for({"example.com": ["93.184.216.34"], "docs.example.com": ["93.184.216.34"]})


def test_rejects_non_http_scheme():
    with pytest.raises(FetchError, match="http"):
        validate_public_url("ftp://example.com/file", resolver=PUBLIC)
    with pytest.raises(FetchError, match="http"):
        validate_public_url("file:///etc/passwd", resolver=PUBLIC)
    with pytest.raises(FetchError, match="http"):
        validate_public_url("javascript:alert(1)", resolver=PUBLIC)


def test_accepts_public_host():
    assert validate_public_url("https://example.com/docs", resolver=PUBLIC) == "https://example.com/docs"


def test_rejects_loopback():
    resolver = resolver_for({"internal": ["127.0.0.1"]})
    with pytest.raises(FetchError, match="private/internal/loopback"):
        validate_public_url("http://internal/", resolver=resolver)


def test_rejects_cloud_metadata_link_local():
    resolver = resolver_for({"metadata": ["169.254.169.254"]})
    with pytest.raises(FetchError, match="private/internal/loopback"):
        validate_public_url("http://metadata/latest/meta-data/", resolver=resolver)


def test_rejects_private_ranges():
    for ip in ["10.0.0.5", "192.168.1.1", "172.16.0.1"]:
        resolver = resolver_for({"host": [ip]})
        with pytest.raises(FetchError):
            validate_public_url("http://host/", resolver=resolver)


def test_rejects_if_any_resolved_address_is_private():
    # A host resolving to both a public and a private address is rejected —
    # not just "use whichever address happens to be public".
    resolver = resolver_for({"mixed": ["93.184.216.34", "127.0.0.1"]})
    with pytest.raises(FetchError):
        validate_public_url("http://mixed/", resolver=resolver)


def test_unresolvable_host_reports_cleanly():
    with pytest.raises(FetchError, match="Cannot resolve"):
        validate_public_url("http://nope.invalid/", resolver=PUBLIC)


def test_html_to_text_strips_script_and_style():
    html_doc = "<html><head><style>.a{}</style></head><body><script>evil()</script><p>Hello</p></body></html>"
    assert html_to_text(html_doc).strip() == "Hello"


def test_html_to_text_unescapes_entities():
    assert "AT&T" in html_to_text("<p>AT&amp;T</p>")


def test_html_to_text_collapses_blank_lines():
    out = html_to_text("<p>a</p>\n\n\n\n<p>b</p>")
    assert "\n\n\n" not in out


class FakeResponse:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.encoding = "utf-8"


def test_fetch_url_success():
    resp = FakeResponse(200, b"hello world", {"content-type": "text/plain"})
    result = fetch_url("http://example.com/", get=lambda u: resp, resolver=PUBLIC)
    assert result == "hello world"


def test_fetch_url_converts_html():
    resp = FakeResponse(200, b"<p>Hello <b>World</b></p>", {"content-type": "text/html; charset=utf-8"})
    result = fetch_url("http://example.com/", get=lambda u: resp, resolver=PUBLIC)
    assert result == "Hello World"


def test_fetch_url_raises_on_http_error():
    resp = FakeResponse(404, b"not found")
    with pytest.raises(FetchError, match="404"):
        fetch_url("http://example.com/", get=lambda u: resp, resolver=PUBLIC)


def test_fetch_url_follows_redirect_to_public_host():
    calls = []

    def get(url):
        calls.append(url)
        if url == "http://example.com/":
            return FakeResponse(302, b"", {"location": "https://docs.example.com/final"})
        return FakeResponse(200, b"final content", {"content-type": "text/plain"})

    result = fetch_url("http://example.com/", get=get, resolver=PUBLIC)
    assert result == "final content"
    assert calls == ["http://example.com/", "https://docs.example.com/final"]


def test_fetch_url_rejects_redirect_to_private_address():
    private_resolver = resolver_for({"example.com": ["93.184.216.34"], "internal.local": ["127.0.0.1"]})

    def get(url):
        if url == "http://example.com/":
            return FakeResponse(302, b"", {"location": "http://internal.local/secrets"})
        raise AssertionError("should never reach the redirect target")

    with pytest.raises(FetchError, match="private/internal/loopback"):
        fetch_url("http://example.com/", get=get, resolver=private_resolver)


def test_fetch_url_too_many_redirects():
    def get(url):
        return FakeResponse(302, b"", {"location": "http://example.com/next"})

    with pytest.raises(FetchError, match="Too many redirects"):
        fetch_url("http://example.com/", get=get, resolver=PUBLIC)


def test_fetch_url_truncates_long_content():
    big = b"x" * 30_000
    resp = FakeResponse(200, big, {"content-type": "text/plain"})
    result = fetch_url("http://example.com/", get=lambda u: resp, resolver=PUBLIC)
    assert len(result) < 30_000
    assert "truncated" in result


def test_fetch_url_empty_response():
    resp = FakeResponse(200, b"", {"content-type": "text/plain"})
    result = fetch_url("http://example.com/", get=lambda u: resp, resolver=PUBLIC)
    assert result == "(empty response)"


def test_fetch_url_wraps_transport_errors():
    def get(url):
        raise ConnectionError("boom")

    with pytest.raises(FetchError, match="boom"):
        fetch_url("http://example.com/", get=get, resolver=PUBLIC)
