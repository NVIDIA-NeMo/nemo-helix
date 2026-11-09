# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin result ignore patterns to Git's own work-tree walk.

Expected results come from ``git ls-files --others --exclude-standard``, which applies
directory pruning and re-inclusion the way Git does. ``git check-ignore`` is not used:
it reports ``a/`` as ignored for ``a/*`` and does not apply parent exclusion.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from nemo_helix_plugin.jobs.ignore_patterns import compile_ignore_rules

CASES: list[tuple[list[str], list[str]]] = [
    (["*", "!*/", "!**/*.txt"], ["nested/a.txt", "nested/b.bin", "a.txt"]),
    (["*", "!plugins/", "!plugins/keep.txt"], ["plugins/keep.txt", "plugins/x"]),
    (["plugins/", "!plugins/keep.txt"], ["plugins/keep.txt", "plugins/x"]),
    (["a/*", "!a/keep.txt"], ["a/keep.txt", "a/other", "a/b/c"]),
    (["a/**", "!a/keep.txt"], ["a/keep.txt", "a/other", "a/b/c"]),
    (["a/*"], ["a/x", "a/b/c", "b/a/x"]),
    (["*.tmp", "!*.tmp/"], ["logs.tmp/keep.txt", "logs.tmp/cache.tmp", "a.tmp"]),
    (["*.tmp"], ["logs.tmp/keep.txt", "a.tmp", "keep.txt"]),
    (["**/nested/*.tmp"], ["x/y/nested/a.tmp", "nested/a.tmp", "nested/b/a.tmp"]),
    (["nested/*.tmp"], ["x/nested/a.tmp", "nested/a.tmp"]),
    (["plugins*"], ["plugins.sha", "plugins/c.json", "n/plugins-x/c", "keep"]),
    (["plugins*/"], ["plugins.sha", "plugins/c.json", "n/plugins-x/c"]),
    (["/evidence/"], ["x/evidence/a", "evidence/a"]),
    (["home/.tmp/plugins*/"], ["home/.tmp/plugins-1/c", "w/home/.tmp/plugins/c", "home/.tmp/plugins.sha"]),
    (["[[:digit:]].tmp", "[^a].x", "\\!y", "\\#z"], ["1.tmp", "q.tmp", "a.x", "b.x", "!y", "#z"]),
    (["a/**", "c/**/b"], ["a/q/r", "c/b", "c/m/n/b", "d/a/q"]),
    (["#cache", "cache.tmp   ", "sp\\ "], ["#cache", "cache", "cache.tmp", "sp "]),
    (["cache.db", "cache/"], ["cache.db", "s/cache.db", "cache/m.bin", "f/cache/m", "metrics.json"]),
]


def _git_excluded(tmp_path: Path, patterns: list[str], files: list[str]) -> set[str]:
    """Return the files Git leaves out of a work-tree walk with ``patterns`` as ``.gitignore``."""
    # Isolate from the caller's Git context (hooks set GIT_DIR/GIT_INDEX_FILE) and global excludes.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    (tmp_path / ".gitignore").write_text("\n".join(patterns) + "\n")
    for relative in files:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(relative)
    listed = subprocess.run(
        ["git", "-C", str(tmp_path), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.split("\0")
    return set(files) - set(listed)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required as the reference implementation")
@pytest.mark.parametrize(("patterns", "files"), CASES)
def test_rules_match_git(tmp_path: Path, patterns: list[str], files: list[str]) -> None:
    rules = compile_ignore_rules(patterns)
    assert rules is not None

    excluded = {relative for relative in files if rules.excludes(PurePosixPath(relative))}

    assert excluded == _git_excluded(tmp_path, patterns, files)


@pytest.mark.parametrize("patterns", [None, [], ""])
def test_empty_patterns_compile_to_none(patterns: list[str] | str | None) -> None:
    assert compile_ignore_rules(patterns) is None


def test_single_string_is_one_pattern() -> None:
    rules = compile_ignore_rules("*.tmp")

    assert rules is not None
    assert rules.excludes(PurePosixPath("nested/a.tmp"))
    assert not rules.excludes(PurePosixPath("a.txt"))


@pytest.mark.parametrize("pattern", ["keep\n*", "keep\r*"])
def test_multiline_pattern_is_rejected(pattern: str) -> None:
    with pytest.raises(ValueError, match="single lines"):
        compile_ignore_rules([pattern])


def test_directory_checks_are_cached(mocker) -> None:
    """Each distinct directory is checked for pruning once; each file is matched once."""
    rules = compile_ignore_rules(["*.tmp"])
    assert rules is not None
    prune = mocker.spy(rules._filter, "may_prune_directory")
    match = mocker.spy(rules._filter, "is_ignored")
    files = [PurePosixPath("a/b/c", name) for name in ("x.txt", "y.txt", "z.tmp")]

    for directory in ("a", "a/b", "a/b/c"):
        rules.excludes(PurePosixPath(directory), is_directory=True)
    for path in files:
        rules.excludes(path)

    assert prune.call_count <= 3
    assert match.call_count <= len(files)
