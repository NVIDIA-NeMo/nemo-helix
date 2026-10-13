# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git worktree helpers for isolated experiment branches."""

import asyncio
import shutil
from pathlib import Path


async def _force_remove_path(path: Path) -> None:
    """Best-effort recursive removal that handles root-owned files.

    The harbor agent runs as root inside its container and writes session
    files (under jobs/) with restricted perms back to the host mount. A
    plain rmtree can't remove those without privilege, so we fall back to
    a one-shot docker container that does have root.
    """
    if not path.exists():
        return

    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return

    proc = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "-v",
        f"{path.parent}:/clean",
        "ubuntu:24.04",
        "rm",
        "-rf",
        f"/clean/{path.name}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


async def create_worktree(
    project_root: Path,
    branch_name: str,
    worktree_path: Path | None = None,
) -> Path:
    """Create a git worktree for isolated changes.

    Args:
        project_root: Path to the main repo.
        branch_name: Name for the new branch.
        worktree_path: Where to create the worktree. Defaults to
            <project_root>/../nhx-worktrees/<branch_name>

    Returns:
        Path to the created worktree.
    """
    if worktree_path is None:
        worktrees_dir = project_root.parent / "nhx-worktrees"
        worktrees_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = worktrees_dir / branch_name

    # If a leftover from a prior crashed/partially-cleaned run sits at the
    # target path, git worktree add fails fatally. Clean defensively first.
    await _force_remove_path(worktree_path)

    proc = await asyncio.create_subprocess_exec(
        "git",
        "worktree",
        "add",
        "-b",
        branch_name,
        str(worktree_path),
        "HEAD",
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {stderr.decode()}")

    return worktree_path


async def remove_worktree(
    project_root: Path,
    worktree_path: Path,
    delete_branch: str | None = None,
) -> None:
    """Remove a git worktree and optionally its branch.

    Args:
        project_root: Path to the main repo.
        worktree_path: Path to the worktree to remove.
        delete_branch: If set, also delete this branch after removing the worktree.
    """
    proc = await asyncio.create_subprocess_exec(
        "git",
        "worktree",
        "remove",
        str(worktree_path),
        "--force",
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    # git worktree remove leaves root-owned harbor session files behind on
    # the host (see _force_remove_path docstring). Make sure the directory
    # is actually gone so the next iteration's create_worktree doesn't trip
    # on a leftover path.
    await _force_remove_path(worktree_path)

    if delete_branch:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "branch",
            "-D",
            delete_branch,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
