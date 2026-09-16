# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the nemo.skills entry-point exposure."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path


def test_entry_point_resolves_to_the_skills_dir() -> None:
    """The nemo.skills entry-point must resolve to a skills_dir holding our skills.

    In a source checkout two distributions advertise this name -- the plugin itself
    and the `nemo-helix` wrapper that bundles it -- so the guarantee is that every
    provider agrees on one target, not that only one provider exists.
    """
    eps = [ep for ep in entry_points(group="nemo.skills") if ep.name == "experimentalist"]
    assert eps, "no 'experimentalist' entry-point in the nemo.skills group"
    assert len({ep.value for ep in eps}) == 1, f"providers disagree on the target: {eps}"

    skills_dir = eps[0].load()
    assert callable(skills_dir), f"entry-point did not resolve to a callable: {skills_dir!r}"

    resolved = skills_dir()
    assert isinstance(resolved, Path), f"skills_dir() returned {resolved!r} (not Path)"
    assert (resolved / "terminator").is_dir(), (
        f"expected a 'terminator' subdir under {resolved!r} -- got {list(resolved.iterdir())}"
    )
