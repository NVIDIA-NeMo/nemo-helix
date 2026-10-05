#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lint fenced bash blocks in NeMo skills against the live `nemo --help` surface.

Catches hallucinated CLI flags and subcommands at PR time. Run from inside the
bootstrapped NeMo venv (the `nemo` binary must be on PATH).

Exit codes:
    0  no violations
    1  one or more skills contain bad commands or flags
    2  internal error (missing `nemo` binary, etc.)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FENCED_BASH = re.compile(r"```bash\n(.*?)```", re.DOTALL)
# Match `nemo …`, `./venv/bin/nemo …`, `.venv/bin/nemo …`, `/abs/path/nemo …`,
# optionally preceded by env-var assignments or a `$ ` prompt. Capture from the
# `nemo` token up to any optional shell comment.
NEMO_LINE = re.compile(
    r"""^\s*                             # leading whitespace
        \$?\s*                           # optional shell prompt
        (?:[A-Z_][A-Z0-9_]*=\S+\s+)*     # optional env-var assignments
        (?:[^\s]+/)?                     # optional path prefix (e.g. .venv/bin/)
        (nemo(?=\s|$)[^#\n]*)            # the command itself
    """,
    re.MULTILINE | re.VERBOSE,
)
FLAG = re.compile(r"(--[A-Za-z][A-Za-z0-9._-]*)")
SKIP_PREFIXES = ("#", "export ", "set ", "echo ", "$(", "&&", "||")

SKILL_GLOBS = [
    "packages/nemo_helix_ext/src/nemo_helix_ext/skills/**/SKILL.md",
    ".agents/skills/**/SKILL.md",
    "plugins/*/skills/**/SKILL.md",
    "plugins/*/src/*/skills/**/SKILL.md",
    "sdk/python/nemo-helix/src/nemo_helix/cli/commands/skills/content/**/SKILL.md",
    "packages/*/src/*/.agents/skills/**/SKILL.md",
]


@dataclass
class Violation:
    file: Path
    line: int
    command: str
    issue: str


def find_skill_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in SKILL_GLOBS:
        found.extend(root.glob(pattern))
    # de-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def extract_nemo_commands(text: str) -> list[tuple[int, str]]:
    """Return (line_number, command) tuples for every `nemo ...` line in fenced bash."""
    commands: list[tuple[int, str]] = []
    for match in FENCED_BASH.finditer(text):
        block = match.group(1)
        block_start_line = text.count("\n", 0, match.start()) + 2  # 1-indexed + ```bash line
        for offset, raw in enumerate(block.split("\n")):
            stripped = raw.strip()
            if not stripped or stripped.startswith(SKIP_PREFIXES):
                continue
            for nemo_match in NEMO_LINE.finditer(raw):
                commands.append((block_start_line + offset, nemo_match.group(1).strip()))
    return commands


def parse_chain_and_flags(command: str) -> tuple[list[str], list[str]]:
    """Split `nemo a b c --foo --bar=baz` into (['nemo','a','b','c'], ['--foo','--bar'])."""
    tokens = command.split()
    chain: list[str] = []
    flags: list[str] = []
    for tok in tokens:
        if tok.startswith("--"):
            match = FLAG.match(tok)
            flags.append(match.group(1) if match else tok.split("=")[0])
        elif tok.startswith("-"):
            # short flag, skip
            continue
        elif (
            tok.startswith("`")
            or tok.startswith("$")
            or tok.startswith("'")
            or tok.startswith('"')
            or tok.startswith("<")
        ):
            # variable substitution, quoted arg, or <placeholder>; stop chain growth
            break
        elif "/" in tok or "=" in tok:
            # path or k=v positional, stop chain
            break
        else:
            chain.append(tok)
    return chain, flags


def get_help(chain: list[str], cache: dict[tuple[str, ...], str | None]) -> str | None:
    key = tuple(chain)
    if key in cache:
        return cache[key]
    try:
        result = subprocess.run(
            chain + ["--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0 and "Usage" not in result.stdout and "Usage" not in result.stderr:
            cache[key] = None
            return None
        cache[key] = (result.stdout or "") + "\n" + (result.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        cache[key] = None
    return cache[key]


def lint_command(
    file: Path,
    line: int,
    command: str,
    cache: dict[tuple[str, ...], str | None],
) -> list[Violation]:
    chain, flags = parse_chain_and_flags(command)
    if len(chain) < 2:  # need at least `nemo <subcommand>`
        return []
    violations: list[Violation] = []

    # Walk the chain, finding the deepest valid subcommand path
    valid_depth = 1
    for depth in range(2, len(chain) + 1):
        help_text = get_help(chain[:depth], cache)
        if help_text is None:
            violations.append(
                Violation(
                    file=file,
                    line=line,
                    command=command,
                    issue=f"unknown subcommand: `{' '.join(chain[:depth])}`",
                )
            )
            return violations
        valid_depth = depth

    # Check each flag against the deepest valid help
    help_text = get_help(chain[:valid_depth], cache) or ""
    for flag in flags:
        if flag not in help_text:
            violations.append(
                Violation(
                    file=file,
                    line=line,
                    command=command,
                    issue=f"unknown flag `{flag}` on `{' '.join(chain[:valid_depth])}`",
                )
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repo root (default: cwd)")
    parser.add_argument("--format", choices=["human", "github"], default="human")
    args = parser.parse_args()

    if not shutil.which("nemo"):
        print(
            "ERROR: `nemo` not on PATH. Run from inside the bootstrapped venv (`source .venv/bin/activate`).",
            file=sys.stderr,
        )
        return 2

    skill_files = find_skill_files(args.root)
    if not skill_files:
        print(f"No skill files found under {args.root}", file=sys.stderr)
        return 2

    cache: dict[tuple[str, ...], str | None] = {}
    all_violations: list[Violation] = []

    for skill in skill_files:
        text = skill.read_text()
        for line, command in extract_nemo_commands(text):
            all_violations.extend(lint_command(skill, line, command, cache))

    if not all_violations:
        print(f"OK: {len(skill_files)} skills, no CLI hallucinations.")
        return 0

    if args.format == "github":
        for v in all_violations:
            print(f"::error file={v.file},line={v.line}::{v.issue}")
    else:
        print(f"Found {len(all_violations)} issues across {len({v.file for v in all_violations})} skills:\n")
        current_file: Path | None = None
        for v in all_violations:
            if v.file != current_file:
                print(f"\n{v.file}")
                current_file = v.file
            print(f"  line {v.line}: {v.issue}")
            print(f"    command: {v.command}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
