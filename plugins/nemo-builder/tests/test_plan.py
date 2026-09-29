# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plan: names first, then where the backend places each image."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import ContextSource


def _spec(name: str, *, repository: str | None = None, tag: str = "v1", fileset: str = "fs-a") -> BuildSpec:
    output = BuildOutput(repository=repository, tag=tag) if repository else None
    return BuildSpec(name=name, source=FileSetSource(fileset=fileset), output=output)


def _set(*specs: BuildSpec, name: str = "demo", revision: int = 3) -> BuildSet:
    return BuildSet(name=name, revision=revision, build_specs=list(specs))


def _one_repository(image: PlannedImage) -> Destination:
    """A stand-in for a backend's rule: every image in one repository, tagged by its row."""
    return Destination(registry="reg.example.com", repository="ws/shared", system_tag=f"t-{image.index}")


class TestResolve:
    def test_names_are_deterministic_from_the_request(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b")), workspace="ws")
        assert plan.job_name == "demo-3"
        assert [image.name for image in plan.images] == ["demo-3-0", "demo-3-1"]
        assert [(image.build_set, image.revision, image.index) for image in plan.images] == [
            ("demo", 3, 0),
            ("demo", 3, 1),
        ]

    def test_nothing_is_placed_until_the_backend_says_where(self) -> None:
        image = BuildPlan.resolve(_set(_spec("a")), workspace="ws").images[0]
        assert image.destination is None
        with pytest.raises(ValueError, match="no destination"):
            _ = image.placed

    def test_an_unqualified_fileset_belongs_to_the_submitting_workspace(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b", fileset="other/fs-b")), workspace="ws")
        assert [image.source.fileset for image in plan.images] == ["ws/fs-a", "other/fs-b"]


class TestWithDestinations:
    def test_every_image_is_placed_by_the_rule(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b")), workspace="ws").with_destinations(_one_repository)
        assert [image.placed.system_ref for image in plan.images] == [
            "reg.example.com/ws/shared:t-0",
            "reg.example.com/ws/shared:t-1",
        ]

    def test_a_spec_without_an_output_has_no_caller_reference(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a")), workspace="ws").with_destinations(_one_repository)
        assert plan.images[0].caller_ref is None

    def test_a_caller_reference_is_on_the_placed_repository(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a", repository="team/app")), workspace="ws")
        placed = plan.with_destinations(_one_repository)
        assert placed.images[0].caller_ref == "reg.example.com/ws/shared:v1"

    def test_two_specs_publishing_one_reference_are_refused(self) -> None:
        """The tag could point at only one of them. A malformed request, so a plain ValueError."""
        plan = BuildPlan.resolve(_set(_spec("a", repository="x"), _spec("b", repository="y")), workspace="ws")
        with pytest.raises(ValueError, match="both publish"):
            plan.with_destinations(_one_repository)

    def test_distinct_tags_on_one_repository_are_fine(self) -> None:
        plan = BuildPlan.resolve(
            _set(_spec("a", repository="x", tag="v1"), _spec("b", repository="x", tag="v2")), workspace="ws"
        )
        assert len(plan.with_destinations(_one_repository).images) == 2


class TestGroups:
    def test_one_group_per_source_in_first_appearance_order(self) -> None:
        plan = BuildPlan.resolve(
            _set(_spec("a", fileset="fs-b"), _spec("b", fileset="fs-a"), _spec("c", fileset="fs-b")),
            workspace="ws",
        )
        groups = plan.groups()
        assert [source for source, _ in groups] == [
            ContextSource(fileset="ws/fs-b"),
            ContextSource(fileset="ws/fs-a"),
        ]
        assert [[image.spec.name for image in images] for _, images in groups] == [["a", "c"], ["b"]]

    def test_context_path_splits_a_shared_fileset(self) -> None:
        """A Dockerfile under `tests/` never sees `environment/`, though both are in one fileset."""
        env = BuildSpec(name="env", source=FileSetSource(fileset="fs-a", context_path="environment"))
        tests = BuildSpec(name="tests", source=FileSetSource(fileset="fs-a", context_path="tests"))
        assert len(BuildPlan.resolve(_set(env, tests), workspace="ws").groups()) == 2

    def test_one_fileset_spelled_two_ways_is_one_group(self) -> None:
        """`fs-a` and `ws/fs-a` are one fileset: qualified, they are one source."""
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b", fileset="ws/fs-a")), workspace="ws")
        assert [source for source, _ in plan.groups()] == [ContextSource(fileset="ws/fs-a")]
