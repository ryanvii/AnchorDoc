#!/usr/bin/env python3
"""Sensitive information scanner to prevent private IPs, keys, or proprietary patterns from entering Git.

Usage:
  python tools/scan_sensitive.py                  # Scan all tracked & untracked worktree files
  python tools/scan_sensitive.py --staged         # Scan git index (pre-commit hook)
  python tools/scan_sensitive.py --history        # Scan full commit history (pre-push check)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUILTIN_RULES = [
    ("private-ip-10", r"(?<![\d.])10\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d])"),
    ("private-ip-172", r"(?<![\d.])172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}(?![\d])"),
    ("private-ip-192", r"(?<![\d.])192\.168\.\d{1,3}\.\d{1,3}(?![\d])"),
    ("api-key-sk", r"\bsk-[A-Za-z0-9_\-]{16,}"),
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("bearer-token", r"Bearer\s+[A-Za-z0-9_\-\.]{24,}"),
    ("aws-key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
]
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".whl"}
SELF_PATH = "tools/scan_sensitive.py"


def load_rules() -> tuple[list[tuple[str, re.Pattern[str]]], int, str]:
    rules = [(name, re.compile(pat)) for name, pat in BUILTIN_RULES]
    local_file = os.environ.get("ANCHORDOC_SCAN_PATTERNS") or str(ROOT / "local" / "scan_patterns.txt")
    n_local = 0
    if Path(local_file).exists():
        for idx, line in enumerate(Path(local_file).read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rules.append((f"local#{idx}", re.compile(s)))
            n_local += 1
    return rules, n_local, local_file


def _mask(s: str) -> str:
    s = s.strip()
    return (s[:2] + "*" * max(0, min(len(s) - 2, 12))) if len(s) > 2 else "**"


def scan_text(name: str, text: str, rules: list[tuple[str, re.Pattern[str]]], hits: list) -> None:
    exempt_builtin = name.endswith(SELF_PATH)
    for ln, line in enumerate(text.splitlines(), 1):
        for rid, rx in rules:
            if exempt_builtin and not rid.startswith("local"):
                continue
            m = rx.search(line)
            if m:
                hits.append((name, ln, rid, _mask(m.group(0))))


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def scan_worktree(rules: list[tuple[str, re.Pattern[str]]], hits: list) -> int:
    files = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    valid = [f for f in files if f and (ROOT / f).is_file()]
    for f in valid:
        p = ROOT / f
        if p.suffix.lower() in SKIP_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        scan_text(f, text, rules, hits)
        scan_text(f"<path> {f}", f, rules, hits)
    return len(valid)


def scan_staged(rules: list[tuple[str, re.Pattern[str]]], hits: list) -> int:
    files = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").decode().split("\0")
    valid = [f for f in files if f]
    for f in valid:
        if Path(f).suffix.lower() in SKIP_EXT:
            continue
        try:
            text = _git("show", f":{f}").decode("utf-8")
        except Exception:
            continue
        scan_text(f, text, rules, hits)
        scan_text(f"<path> {f}", f, rules, hits)
    return len(valid)


def scan_history(rng: list[str], rules: list[tuple[str, re.Pattern[str]]], hits: list) -> tuple[int, int]:
    revs = _git("rev-list", *(rng or ["--all"])).decode().split()
    seen: set[str] = set()
    for rev in revs:
        msg = _git("log", "-1", "--format=%an <%ae>%n%cn <%ce>%n%B", rev).decode("utf-8", "replace")
        scan_text(f"<commit {rev[:8]}>", msg, rules, hits)
        for row in _git("ls-tree", "-r", "-z", rev).decode().split("\0"):
            if not row:
                continue
            meta, path = row.split("\t", 1)
            _, typ, sha = meta.split()
            scan_text(f"<path@{rev[:8]}> {path}", path, rules, hits)
            if typ != "blob" or sha in seen or Path(path).suffix.lower() in SKIP_EXT:
                continue
            seen.add(sha)
            try:
                text = _git("cat-file", "-p", sha).decode("utf-8")
            except Exception:
                continue
            scan_text(f"{path}@{rev[:8]}", text, rules, hits)
    return len(revs), len(seen)


def main(argv: list[str]) -> int:
    rules, n_local, local_file = load_rules()
    hits: list = []
    if "--staged" in argv:
        n = scan_staged(rules, hits)
        label = f"staged ({n} files)"
    elif "--history" in argv:
        rng = [a for a in argv if a != "--history"]
        nc, nb = scan_history(rng, rules, hits)
        label = f"history ({nc} commits / {nb} blobs)"
    else:
        n = scan_worktree(rules, hits)
        label = f"worktree ({n} files)"

    print(f"[scan] {label} | builtin={len(BUILTIN_RULES)}, local={n_local}")
    for name, ln, rid, frag in hits:
        print(f"  HIT {name}:{ln} [{rid}] {frag}")
    print(f"[scan] total hits: {len(hits)}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
