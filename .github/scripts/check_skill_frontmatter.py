#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml>=6.0.2",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that new or modified canonical SKILL.md files carry `metadata.author` and `allowed-tools`.

Both fields are load-bearing per docs/contributing/skills-spec.mdx: `allowed-tools` is the
safety signal the agent harness honors, and `metadata.author` is how a skill maintainer is
found. This only checks files a PR actually touches, so it does not block unrelated work on
skills that predate the rule; it stops the gap from growing.

Exit codes:
    0  no violations
    1  one or more canonical SKILL.md files are missing a required field
    2  internal error (bad frontmatter, git failure, etc.)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Canonical skill locations per docs/contributing/skills-spec.mdx. Internal dev-only skill
# dirs (.agents/skills/, .cursor/rules/, agent-runtime skill bundles under agents/, web/.agents/
# skills/) intentionally use a lighter frontmatter and are out of scope.
CANONICAL_SKILL_GLOBS = [
    "skills/*/SKILL.md",
    "packages/nemo_platform_ext/src/nemo_platform_ext/skills/*/SKILL.md",
    "plugins/*/skills/*/SKILL.md",
    "plugins/*/src/*/skills/*/SKILL.md",
    "plugins/*/framework-skills/*/SKILL.md",
]

REQUIRED_FIELDS = ("metadata.author", "allowed-tools")

# Comma-separated `allowed-tools` entries must be bare tool names (e.g. `Bash`, `mcp__nemo__list`),
# not free text: this catches unparseable values like "Bash Read" (missing comma).
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class MalformedFrontmatterError(Exception):
    """Raised when a SKILL.md's frontmatter block is present but not valid YAML."""


def _glob_to_path_regex(pattern: str) -> re.Pattern[str]:
    # `*` matches one path segment only (unlike fnmatch, which also matches `/`).
    segments = (re.escape(segment).replace(r"\*", "[^/]+") for segment in pattern.split("/"))
    return re.compile("^" + "/".join(segments) + "$")


_CANONICAL_SKILL_PATTERNS = [_glob_to_path_regex(pattern) for pattern in CANONICAL_SKILL_GLOBS]


@dataclass
class Violation:
    file: Path
    issue: str


def is_canonical_skill(path: Path) -> bool:
    posix = path.as_posix()
    return any(pattern.match(posix) for pattern in _CANONICAL_SKILL_PATTERNS)


def changed_skill_files(root: Path, base_ref: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD", "--", "**/SKILL.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def frontmatter(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    block = text[4:end]
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise MalformedFrontmatterError(f"{path}: frontmatter block is not valid YAML: {exc}") from exc
    return data if isinstance(data, dict) else None


def check_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    data = frontmatter(path)
    if data is None:
        return [Violation(path, "file does not open with a valid YAML frontmatter block")]

    metadata = data.get("metadata")
    author = metadata.get("author") if isinstance(metadata, dict) else None
    if not isinstance(author, str) or not author.strip():
        violations.append(Violation(path, "missing or invalid `metadata.author` (must be a non-empty string)"))

    tools = data.get("allowed-tools")
    if not isinstance(tools, str) or not tools.strip():
        violations.append(
            Violation(path, "missing or empty `allowed-tools` (must be a comma-separated string, e.g. Bash, Read)")
        )
    elif not all(_TOOL_NAME_RE.match(entry.strip()) for entry in tools.split(",")):
        violations.append(
            Violation(
                path,
                f"invalid `allowed-tools` {tools!r} (must be a comma-separated list of tool names, e.g. Bash, Read)",
            )
        )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repo root (default: cwd)")
    parser.add_argument("--base-ref", default="origin/main", help="Diff base ref (default: origin/main)")
    parser.add_argument("--format", choices=["human", "github"], default="human")
    parser.add_argument(
        "files", nargs="*", type=Path, help="Explicit SKILL.md paths to check instead of diffing against base-ref"
    )
    args = parser.parse_args()

    if args.files:
        candidates = [args.root / f if not f.is_absolute() else f for f in args.files]
    else:
        try:
            candidates = changed_skill_files(args.root, args.base_ref)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: git diff against {args.base_ref!r} failed: {exc.stderr}", file=sys.stderr)
            return 2

    skill_files = []
    for f in candidates:
        if not f.exists():
            continue
        try:
            rel = f.relative_to(args.root)
        except ValueError:
            continue
        if is_canonical_skill(rel):
            skill_files.append(f)

    if not skill_files:
        print("OK: no new or modified canonical SKILL.md files.")
        return 0

    all_violations: list[Violation] = []
    try:
        for skill in sorted(skill_files):
            all_violations.extend(check_file(skill))
    except MalformedFrontmatterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not all_violations:
        print(f"OK: {len(skill_files)} changed skill(s), {' and '.join(REQUIRED_FIELDS)} present.")
        return 0

    if args.format == "github":
        for v in all_violations:
            try:
                rel = v.file.relative_to(args.root)
            except ValueError:
                rel = v.file
            print(f"::error file={rel}::{v.issue}")
    else:
        print(f"Found {len(all_violations)} issues across {len({v.file for v in all_violations})} skills:\n")
        current_file: Path | None = None
        for v in all_violations:
            if v.file != current_file:
                print(f"\n{v.file}")
                current_file = v.file
            print(f"  {v.issue}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
