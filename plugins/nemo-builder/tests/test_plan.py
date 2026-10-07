# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plan: names first, then where the backend places each image."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import ContextSource


def _spec(name: str | None, *, repository: str | None = None, tag: str = "v1", fileset: str = "fs-a") -> BuildSpec:
    output = BuildOutput(repository=repository, tag=tag) if repository else None
    return BuildSpec(name=name, source=FileSetSource(fileset=fileset), output=output)


def _set(*specs: BuildSpec, name: str = "demo", revision: int = 3) -> BuildSet:
    return BuildSet(name=name, revision=revision, build_specs=list(specs))


def _one_repository(image: PlannedImage) -> Destination:
    return Destination(registry="reg.example.com", repository="ws/shared", system_tag=f"t-{image.name}")


class TestResolve:
    def test_names_are_deterministic_from_the_request(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b")), workspace="ws")
        assert plan.job_name == "demo-3"
        assert [image.name for image in plan.images] == ["demo-3.a", "demo-3.b"]

    def test_an_unnamed_spec_is_named_after_the_set(self) -> None:
        plan = BuildPlan.resolve(_set(_spec(None), _spec("tests")), workspace="ws")
        assert [image.name for image in plan.images] == ["demo-3", "demo-3.tests"]

    def test_names_do_not_depend_on_the_order_of_the_specs(self) -> None:
        forward = BuildPlan.resolve(_set(_spec("a"), _spec("b")), workspace="ws")
        backward = BuildPlan.resolve(_set(_spec("b"), _spec("a")), workspace="ws")
        assert {image.name for image in forward.images} == {image.name for image in backward.images}

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
            "reg.example.com/ws/shared:t-demo-3.a",
            "reg.example.com/ws/shared:t-demo-3.b",
        ]

    def test_a_spec_without_an_output_has_no_caller_reference(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a")), workspace="ws").with_destinations(_one_repository)
        assert plan.images[0].caller_ref is None

    def test_a_caller_reference_is_on_the_placed_repository(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a", repository="team/app")), workspace="ws")
        placed = plan.with_destinations(_one_repository)
        assert placed.images[0].caller_ref == "reg.example.com/ws/shared:v1"

    def test_two_specs_publishing_one_reference_are_refused(self) -> None:
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
        env = BuildSpec(name="env", source=FileSetSource(fileset="fs-a", context_path="environment"))
        tests = BuildSpec(name="tests", source=FileSetSource(fileset="fs-a", context_path="tests"))
        assert len(BuildPlan.resolve(_set(env, tests), workspace="ws").groups()) == 2

    def test_one_fileset_spelled_two_ways_is_one_group(self) -> None:
        plan = BuildPlan.resolve(_set(_spec("a"), _spec("b", fileset="ws/fs-a")), workspace="ws")
        assert [source for source, _ in plan.groups()] == [ContextSource(fileset="ws/fs-a")]

    def test_an_archive_and_each_directory_in_it_is_its_own_group(self) -> None:
        def task(name: str, context_path: str | None) -> BuildSpec:
            source = FileSetSource(fileset="fs-a", archive="task.tar.gz", context_path=context_path)
            return BuildSpec(name=name, source=source)

        plan = BuildPlan.resolve(
            _set(task("env", "environment"), task("env-again", "environment"), task("whole", None), _spec("plain")),
            workspace="ws",
        )
        assert [source for source, _ in plan.groups()] == [
            ContextSource(fileset="ws/fs-a", archive="task.tar.gz", context_path="environment"),
            ContextSource(fileset="ws/fs-a", archive="task.tar.gz"),
            ContextSource(fileset="ws/fs-a"),
        ]
