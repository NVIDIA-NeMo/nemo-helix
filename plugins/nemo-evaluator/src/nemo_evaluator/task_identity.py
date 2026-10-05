# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared task selector qualification, identity checks, and compatibility errors."""

from nemo_evaluator.api.fields import TaskRef, parse_subentity_ref


class UnsupportedTaskKindError(ValueError):
    """A stored task cannot run on the requested target."""


def qualified_task_refs(refs: list[TaskRef], workspace: str) -> list[TaskRef]:
    """Rewrite each ref to ``workspace/name#fragment`` before any store access.

    Algorithm:
        - Fill a missing workspace from ``workspace`` and a missing fragment with ``latest``.
        - Preserve explicit selectors while rejecting repeated ``workspace/name`` identities.
        - Reject an empty collection so downstream resolution always has at least one task.
    """
    seen: set[tuple[str, str]] = set()
    qualified = []
    for ref in refs:
        member_workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
        identity = (member_workspace, name)
        if identity in seen:
            raise ValueError(f"Duplicate task identity: {member_workspace}/{name}")
        seen.add(identity)
        qualified.append(TaskRef(f"{member_workspace}/{name}#{fragment}"))
    if not qualified:
        raise ValueError("Expected at least one stored task")
    return qualified
