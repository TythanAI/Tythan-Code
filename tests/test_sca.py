"""SCA (dependency vulnerability) tests. Offline — OSV is mocked."""

from tythancode.sca import (
    Dependency,
    collect_dependencies,
    parse_package_json,
    parse_pyproject,
    parse_requirements,
    query_osv,
    scan_dependencies,
)


def test_parse_requirements():
    text = (
        "requests==2.19.0\n"
        "flask>=2.0\n"           # not pinned — skipped
        "# comment\n"
        "Django==3.2.1  # pinned\n"
    )
    deps = parse_requirements(text, "requirements.txt")
    assert [(d.name, d.version) for d in deps] == [("requests", "2.19.0"), ("django", "3.2.1")]
    assert all(d.ecosystem == "PyPI" for d in deps)


def test_parse_pyproject():
    text = (
        "[project]\n"
        'dependencies = [\n'
        '    "httpx==0.27.0",\n'
        '    "rich>=13.0",\n'
        ']\n'
    )
    deps = parse_pyproject(text, "pyproject.toml")
    assert [(d.name, d.version) for d in deps] == [("httpx", "0.27.0")]


def test_parse_package_json():
    text = '{"dependencies": {"lodash": "4.17.20", "react": "^18.0.0"}, "devDependencies": {"jest": "~29.0.0"}}'
    deps = parse_package_json(text, "package.json")
    names = {(d.name, d.version) for d in deps}
    assert ("lodash", "4.17.20") in names
    assert ("react", "18.0.0") in names  # ^ prefix stripped
    assert ("jest", "29.0.0") in names


def test_collect_dependencies_skips_junk(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests==2.19.0\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package.json").write_text('{"dependencies": {"x": "1.0.0"}}')
    deps = collect_dependencies(tmp_path)
    assert len(deps) == 1
    assert deps[0].name == "requests"


def test_query_osv_maps_findings():
    deps = [
        Dependency("PyPI", "requests", "2.19.0", "requirements.txt"),
        Dependency("PyPI", "safe-pkg", "1.0.0", "requirements.txt"),
    ]

    def fake_post(url, payload):
        assert len(payload["queries"]) == 2
        return {"results": [
            {"vulns": [{"id": "GHSA-x84v"}, {"id": "CVE-2018-18074"}]},
            {},
        ]}

    findings = query_osv(deps, post=fake_post)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule == "SCA-PYPI"
    assert "requests 2.19.0" in f.message
    assert "GHSA-x84v" in f.message
    assert f.path == "requirements.txt"


def test_scan_dependencies_offline_degrades(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests==2.19.0\n")

    def broken_post(url, payload):
        raise ConnectionError("no network")

    findings, note = scan_dependencies(tmp_path, post=broken_post)
    assert findings == []
    assert "skipped" in note


def test_scan_dependencies_no_pins(tmp_path):
    (tmp_path / "requirements.txt").write_text("flask\n")
    findings, note = scan_dependencies(tmp_path, post=lambda u, p: {"results": []})
    assert findings == []
    assert "no pinned dependencies" in note
