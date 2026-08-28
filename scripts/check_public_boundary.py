"""Fail closed when a tracked public file crosses the repository safety boundary."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ALLOWED_SUFFIXES = {
    "",
    ".cff",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BLOCKED_SUFFIXES = {
    ".7z",
    ".csv",
    ".db",
    ".gz",
    ".jpeg",
    ".jpg",
    ".key",
    ".parquet",
    ".p12",
    ".pem",
    ".pfx",
    ".png",
    ".sqlite",
    ".zip",
}
SECRET_PATTERNS = {
    "private-key-marker": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "aws-access-key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "openai-token": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
}


def candidate_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = tuple(
        root / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )
    return tuple(sorted(paths))


def audit_file(root: Path, path: Path) -> tuple[str, ...]:
    relative = path.relative_to(root)
    violations: list[str] = []
    lowered_parts = {part.lower() for part in relative.parts}
    lowered_name = relative.name.lower()
    suffix = relative.suffix.lower()
    if ".secrets" in lowered_parts or lowered_name.startswith(".env"):
        violations.append("blocked-secret-path")
    if suffix in BLOCKED_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        violations.append("blocked-file-type")
    content = path.read_bytes()
    if b"\0" in content:
        violations.append("binary-content")
        return tuple(violations)
    for code, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            violations.append(code)
    return tuple(violations)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations: list[tuple[str, str]] = []
    files = candidate_files(root)
    for path in files:
        for code in audit_file(root, path):
            violations.append((path.relative_to(root).as_posix(), code))
    print(f"checked_file_count={len(files)}")
    print(f"violation_count={len(violations)}")
    for relative_path, code in violations:
        print(f"violation={code} path={relative_path}")
    if violations:
        print("boundary_status=failed")
        return 1
    print("boundary_status=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
