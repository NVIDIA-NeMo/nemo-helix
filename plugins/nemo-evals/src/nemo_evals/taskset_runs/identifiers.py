# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable caller-scoped run names and per-task record/Job names."""

from hashlib import sha256


def derive_run_id(workspace: str, caller: str, idempotency_key: str) -> str:
    """Return a 20-character run name scoped to workspace, caller and key.

    A key, including an empty one, keeps its name until the run record is deleted. Submit generates
    a key when the caller omits one.

    Args:
        workspace: Namespace that isolates runs belonging to different workspaces.
        caller: Authenticated identity that isolates callers reusing the same key.
        idempotency_key: Logical retry key, the caller's or one submit generated.

    Returns:
        ``run-`` plus 16 hex characters. Names hash null-separated scope
        components so retries find the same named-create acceptance point.
    """
    digest = sha256(f"{workspace}\0{caller}\0{idempotency_key}".encode("utf-8")).hexdigest()
    return f"run-{digest[:16]}"


def member_name(task_index: int) -> str:
    """Return the member name task-N for a resolved task's zero-based index."""
    return f"task-{task_index}"


def task_job_name(run_id: str, task_index: int) -> str:
    """Return the task Job name for run_id and its zero-based task index."""
    return f"{run_id}-{member_name(task_index)}"


def bundle_job_name(run_id: str, attempt: int) -> str:
    """Return the name of the run's ``attempt``-th bundle Job (1-based): ``<run_id>-bundle-<attempt>``."""
    return f"{run_id}-bundle-{attempt}"
