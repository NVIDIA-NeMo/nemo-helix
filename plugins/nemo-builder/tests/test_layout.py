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
        assert layout.context(ContextSource(fileset="ws/fs-a")) == ROOT / "context/ws/fs-a"

    def test_a_subtree_sits_inside_its_fileset(self) -> None:
        """What lets one whole-fileset download serve every subtree of it."""
        subtree = layout.context(ContextSource(fileset="ws/fs-a", context_path="env/tests"))
        assert subtree == ROOT / "context/ws/fs-a/env/tests"
        assert layout.fileset("ws/fs-a") in subtree.parents

    def test_a_cross_workspace_fileset_nests_under_its_workspace(self) -> None:
        assert layout.fileset("other-ws/fs-a") == ROOT / "context/other-ws/fs-a"

    def test_the_context_hash_is_outside_every_context(self) -> None:
        """Inside any context, it would become part of what some Dockerfile builds from."""
        whole = ContextSource(fileset="ws/fs-a")
        subtree = ContextSource(fileset="ws/fs-a", context_path="tests")
        for source in (whole, subtree):
            hash_file = layout.context_hash_file(source)
            assert ROOT / "context" not in hash_file.parents
        assert layout.context_hash_file(whole) != layout.context_hash_file(subtree)

    def test_outputs_are_one_directory_per_image(self) -> None:
        assert layout.output("demo-1-0") == layout.outputs / "demo-1-0" == ROOT / "out/demo-1-0"

    def test_the_same_names_give_the_same_relative_paths_under_any_root(self) -> None:
        """The property every step relies on: fetch, supervise and push root the layout
        differently, and agree on everything beneath the root."""
        source = ContextSource(fileset="ws/fs-a", context_path="tests")
        roots = [PurePosixPath("/var/run/scratch/job"), PurePosixPath("jobs/default/abc"), PurePosixPath("/nhx-work")]
        relative = {
            (WorkLayout(r).context(source).relative_to(r), WorkLayout(r).output("x").relative_to(r)) for r in roots
        }
        assert len(relative) == 1


class TestFilesetsNeverNest:
    """`foo` in one's own workspace and `foo/bar` in workspace `foo` would otherwise be one
    directory inside the other, and the first context would hold the second's files."""

    def test_an_unqualified_fileset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="qualified"):
            ContextSource(fileset="foo")

    def test_a_workspace_named_like_a_fileset_does_not_nest_inside_it(self) -> None:
        own = layout.fileset("ws/foo")
        other = layout.fileset("foo/bar")
        assert own not in other.parents and other not in own.parents

    @pytest.mark.parametrize("fileset", ["foo", "a/b/c"])
    def test_the_layout_refuses_anything_but_two_components(self, fileset: str) -> None:
        with pytest.raises(ValueError, match="qualified"):
            layout.fileset(fileset)


class TestNothingLeavesTheJobDirectory:
    """`PurePosixPath("a") / "/etc"` is `/etc`. A caller-supplied component that is absolute, or
    that climbs, would point a step outside this job's slice of a volume every build shares."""

    @pytest.mark.parametrize("fileset", ["/etc/x", "../other-job", "ws/..", ""])
    def test_a_fileset_name_that_escapes_is_refused(self, fileset: str) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.fileset(fileset)

    @pytest.mark.parametrize("context_path", ["/etc", "../fs-b", "tests/../../.."])
    def test_a_context_path_that_escapes_is_refused(self, context_path: str) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.context(ContextSource(fileset="ws/fs-a", context_path=context_path))

    def test_an_image_name_that_escapes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a relative path"):
            layout.output("../context/fs-a")
