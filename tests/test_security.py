"""Security scanner tests. Offline."""

from tythancode.security import format_findings, scan_text, scan_workspace
from tythancode.tools import Workspace


def rules_found(findings):
    return {f.rule for f in findings}


def test_detects_aws_key():
    findings = scan_text('aws_key = "AKIAIOSFODNN7REALKEY"', "cfg.py")
    assert "SEC-AWS-KEY" in rules_found(findings)
    assert findings[0].severity == "CRITICAL"
    assert findings[0].line == 1


def test_detects_private_key_and_hardcoded_cred():
    text = '-----BEGIN RSA PRIVATE KEY-----\npassword = "hunter2hunter2"\n'
    rules = rules_found(scan_text(text, "secrets.txt"))
    assert "SEC-PRIVATE-KEY" in rules
    assert "SEC-HARDCODED-CRED" in rules


def test_fixture_lines_downgraded():
    findings = scan_text('key = "AKIAIOSFODNN7EXAMPLE"  # example', "docs.py")
    aws = [f for f in findings if f.rule == "SEC-AWS-KEY"]
    assert aws and aws[0].severity == "MEDIUM"


def test_detects_dangerous_python():
    text = (
        "import pickle, yaml, subprocess\n"
        "data = pickle.loads(blob)\n"
        "cfg = yaml.load(f)\n"
        "subprocess.run(cmd, shell=True)\n"
        'cur.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        "requests.get(url, verify=False)\n"
    )
    rules = rules_found(scan_text(text, "app.py"))
    assert {"PY-PICKLE", "PY-YAML-LOAD", "PY-SHELL-TRUE", "PY-SQL-FSTRING", "PY-VERIFY-FALSE"} <= rules


def test_yaml_safe_loader_ok():
    findings = scan_text("yaml.load(f, Loader=yaml.SafeLoader)", "app.py")
    assert "PY-YAML-LOAD" not in rules_found(findings)


def test_clean_code_no_findings():
    text = "import os\nkey = os.environ['API_KEY']\nprint('hello')\n"
    assert scan_text(text, "clean.py") == []


def test_scan_workspace_and_ordering(tmp_path):
    (tmp_path / "a.py").write_text('token = "AKIAIOSFODNN7REALKEY"\n')
    (tmp_path / "b.py").write_text("x = yaml.load(f)\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "leaked.py").write_text('p = "AKIAIOSFODNN7REALKEY"\n')
    (tmp_path / "image.png").write_bytes(b"AKIAIOSFODNN7REALKEY")

    findings = scan_workspace(Workspace(tmp_path))
    paths = [f.path for f in findings]
    assert "a.py" in paths and "b.py" in paths
    assert not any(".git" in p or p.endswith(".png") for p in paths)
    # CRITICAL sorts before MEDIUM
    assert findings[0].severity == "CRITICAL"


def test_scan_single_file(tmp_path):
    (tmp_path / "ok.py").write_text("print('hi')\n")
    (tmp_path / "bad.py").write_text("eval(user_input)\n")
    findings = scan_workspace(Workspace(tmp_path), "bad.py")
    assert rules_found(findings) == {"PY-EVAL"}


def test_new_secret_rules():
    text = (
        'stripe = "sk_live_' + "a1B2c3D4e5F6g7H8i9J0k1L2" + '"\n'
        'g = "AIza' + "SyA" + "x" * 32 + '"\n'
        'auth = {"Authorization": "Bearer abcdefghij1234567890XYZ"}\n'
    )
    rules = rules_found(scan_text(text, "keys.py"))
    assert {"SEC-STRIPE-KEY", "SEC-GOOGLE-KEY", "SEC-BEARER-TOKEN"} <= rules


def test_new_code_rules():
    text = (
        "cipher = AES.new(key, AES.MODE_ECB)\n"
        "token = str(random.randint(0, 999999))\n"
        "path = tempfile.mktemp()\n"
        'url = "http://api.example.com/v1"\n'
        'safe = "http://localhost:8080"\n'
    )
    rules = rules_found(scan_text(text, "app.py"))
    assert {"PY-WEAK-CIPHER", "PY-INSECURE-RANDOM", "PY-MKTEMP", "NET-PLAIN-HTTP"} <= rules
    # localhost http is not flagged
    assert sum(1 for f in scan_text(text, "app.py") if f.rule == "NET-PLAIN-HTTP") == 1


def test_entropy_detection():
    import secrets as pysecrets

    random_key = pysecrets.token_urlsafe(32)
    findings = scan_text(f'db_key = "{random_key}"\n', "cfg.py")
    assert "SEC-HIGH-ENTROPY" in rules_found(findings)

    # low-entropy and non-secret contexts don't fire
    assert "SEC-HIGH-ENTROPY" not in rules_found(
        scan_text('key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n', "cfg.py"))
    assert "SEC-HIGH-ENTROPY" not in rules_found(
        scan_text(f'greeting = "{random_key}"\n', "cfg.py"))


def test_entropy_skipped_when_specific_rule_fires():
    # 40-char GitHub token: entropy candidate, but the specific rule wins
    findings = scan_text('token = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"\n', "cfg.py")
    rules = rules_found(findings)
    assert "SEC-GITHUB-TOKEN" in rules
    assert "SEC-HIGH-ENTROPY" not in rules


def test_lockfiles_skipped(tmp_path):
    (tmp_path / "package-lock.json").write_text('{"password": "supersecretvalue123"}\n')
    (tmp_path / "app.min.js").write_text('password = "supersecretvalue123"\n')
    assert scan_workspace(Workspace(tmp_path)) == []


def test_format_findings_report(tmp_path):
    (tmp_path / "bad.py").write_text('password = "supersecretvalue"\n')
    report = format_findings(scan_workspace(Workspace(tmp_path)))
    assert "SEC-HARDCODED-CRED" in report
    assert "bad.py:1" in report
    assert "Summary:" in report
    assert "No security findings" in format_findings([])
