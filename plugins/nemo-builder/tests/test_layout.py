# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work-volume layout: one definition, rooted differently by each step."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest
from nemo_builder_plugin.steps import ContextSource, WorkLayout

ROOT = PurePosixPath("/mnt/job")
layout = WorkLayout(ROOT)


class TestWhereThingsLive:
    def test_a_whole_fileset_context_is_its_fileset_directory(self) -> None:
        assert layout.context(ContextSource(fileset="fs-a")) == ROOT / "context/fs-a"

    def test_a_subtree_sits_inside_its_fileset(self) -> None:
        """What lets one whole-fileset download serve every subtree of it."""
        subtree = layout.context(ContextSource(fileset="fs-a", context_path="env/tests"))
        assert subtree == ROOT / "context/fs-a/env/tests"
        assert layout.fileset("fs-a") in subtree.parents

    def test_a_cross_workspace_fileset_nests_under_its_workspace(self) -> None:
        assert layout.fileset("other-ws/fs-a") == ROOT / "context/other-ws/fs-a"

    def test_the_context_hash_is_beside_the_context_not_in_it(self) -> None:
        """Inside, it would become part of what the Dockerfile builds from."""
        source = ContextSource(fileset="fs-a", context_path="tests")
        hash_file = layout.context_hash_file(source)
        assert hash_file.parent == layout.context(source).parent
        assert layout.context(source) not in hash_file.parents

    def test_outputs_are_one_directory_per_image(self) -> None:
        assert layout.output("demo-1-0") == layout.outputs / "demo-1-0" == ROOT / "out/demo-1-0"

    def test_the_same_names_give_the_same_relative_paths_under_any_root(self) -> None:
        """The property every step relies on: fetch, supervise and push root the layout
        differently, and agree on everything beneath the root."""
        source = ContextSource(fileset="fs-a", context_path="tests")
        roots = [PurePosixPath("/var/run/scratch/job"), PurePosixPath("jobs/default/abc"), PurePosixPath("/nhx-work")]
        relative = {
            (WorkLayout(r).context(source).relative_to(r), WorkLayout(r).output("x").relative_to(r)) for r in roots
        }
        assert len(relative) == 1


class TestNothingLeavesTheJobDirectory:
    """`PurePosixPath("a") / "/etc"` is `/etc`. A caller-supplied component that is absolute, or
    that climbs, would point a step outside this job's slice of a volume every build shares."""

    @pytest.mark.parametrize("fileset", ["/etc", "../other-job", "fs/../../x", ""])
    def test_a_fileset_name_that_escapes_is_refused(self, fileset: str) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.context(ContextSource(fileset=fileset))

    @pytest.mark.parametrize("context_path", ["/etc", "../fs-b", "tests/../../.."])
    def test_a_context_path_that_escapes_is_refused(self, context_path: str) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.context(ContextSource(fileset="fs-a", context_path=context_path))

    def test_an_image_name_that_escapes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.output("../context/fs-a")
