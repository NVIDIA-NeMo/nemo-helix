# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry-point lookup shared by the agents plugin's plugin-provided registries.

Post-run actions and plugin-managed agent types are both registered by name in
an entry-point group, and both must tolerate the same name being declared twice
for one implementation (see :func:`load_unique_entry_point`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from importlib.metadata import EntryPoint
from typing import Any


def group_entry_points(found: Iterable[EntryPoint]) -> dict[str, list[EntryPoint]]:
    """Group *found* (one group's entry points) by name.

    Callers pass ``importlib.metadata.entry_points(group=...)`` from their own
    module, so a test can substitute one registry's entry points without
    touching the other's.
    """
    registered: dict[str, list[EntryPoint]] = defaultdict(list)
    for entry_point in found:
        registered[entry_point.name].append(entry_point)
    return registered


def load_unique_entry_point(name: str, registered: dict[str, list[EntryPoint]], *, label: str, plural: str) -> Any:
    """Load the entry point registered as *name*, or raise ``ValueError``.

    *label* and *plural* name what the group holds, e.g. ``"post-run action"``
    and ``"post-run actions"``, for the error messages.
    """
    matches = registered.get(name)
    if not matches:
        installed = ", ".join(sorted(registered)) or "none"
        raise ValueError(f"Unknown {label} {name!r}. Installed {plural}: {installed}.")

    # A name may legitimately be declared by more than one installed
    # distribution: the aggregate ``nemo-helix`` wheel re-declares every
    # bundled plugin's entry points, so a standard install sees each of them
    # twice — once from the plugin, once from the aggregate. What must be
    # unique is the *implementation*, not the number of declarations, so
    # collapse identical targets and reject only genuine conflicts.
    targets = {entry_point.value for entry_point in matches}
    if len(targets) > 1:
        conflicting = ", ".join(sorted(targets))
        raise ValueError(f"Conflicting {plural} are registered for {name!r}: {conflicting}.")
    return matches[0].load()
