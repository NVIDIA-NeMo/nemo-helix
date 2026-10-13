# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClaudeCodingAgent — invokes ``claude --print`` in a worktree.

POC: thin wrapper. The actual subprocess call still lives in ``loop.py`` for
now (apply_hypothesis); this module just exposes the protocol-shaped wrapper
for the v1 codex work to slot in alongside.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from .base import InvocationResult


class ClaudeCodingAgent:
    """Coding agent backed by the Claude Code CLI (``claude --print``)."""

    name = "claude"

    def preflight(self) -> None:
        """Verify the ``claude`` CLI is on PATH."""
        if not shutil.which("claude"):
            raise RuntimeError(
                "Coding agent 'claude' not found on PATH. Install Claude Code "
                "(https://claude.com/claude-code) and authenticate before running "
                "the optimize-skills loop."
            )

    async def invoke(self, prompt: str, worktree: Path, *, timeout: float = 600.0) -> InvocationResult:
        # Strip ANTHROPIC_* so the CLI uses OAuth instead of picking up API
        # keys intended for the NeMo Helix application. Also strip CLAUDE_CODE_* /
        # CLAUDECODE so the subprocess doesn't refuse to nest when the loop
        # is launched from inside an active Claude Code session.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("ANTHROPIC_") and k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")
        }

        proc = await asyncio.create_subprocess_exec(
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=str(worktree),
            env=clean_env,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(input=prompt.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            # Drain pending output so pipes close and the child is reaped —
            # otherwise repeated timeouts leak zombies across iterations.
            try:
                await proc.communicate()
            except Exception:
                pass
            return InvocationResult(returncode=-1)

        # Determine changed files via git
        changed = await _git_changed_files(worktree)
        return InvocationResult(
            changed_files=changed,
            explanation=stdout.decode() if stdout else "",
            returncode=proc.returncode or 0,
        )


async def _git_changed_files(worktree: Path) -> list[str]:
    """All files that differ from HEAD — staged, unstaged, untracked, deleted, renamed.

    `git status --porcelain` is the single source of truth here: `git diff
    --name-only` alone misses staged edits, deletions, and renames, and the
    untracked-only fallback misses staged adds.
    """
    proc = await asyncio.create_subprocess_exec(
        "git", "status", "--porcelain", cwd=str(worktree), stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    files: set[str] = set()
    for line in out.decode().splitlines():
        if not line:
            continue
        # Porcelain v1: "XY <path>" or rename "R  <old> -> <new>". Take the
        # post-rename path so callers see the file at its current location.
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.add(path)
    return sorted(files)
