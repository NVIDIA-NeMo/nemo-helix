# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema rules, and the entity's write-once shape."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import MAX_BUILD_SPECS, BuildOutput, BuildSet, BuildSpec, FileSetSource
from pydantic import ValidationError

_DIGEST = "sha256:" + "b" * 64


def _spec(
    name: str, *, fileset: str = "fs-a", context_path: str | None = None, dockerfile: str = "Dockerfile"
) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(repository=f"team/{name}", tag="v1"),
        dockerfile=dockerfile,
    )


class TestDockerfileStaysWithinContext:
    @pytest.mark.parametrize("good", ["Dockerfile", "event_feed/Dockerfile", "a/b/c.dockerfile"])
    def test_descends(self, good: str) -> None:
        assert _spec("x", dockerfile=good).dockerfile == good

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../Dockerfile", "a/../../b/Dockerfile", "", "."])
    def test_does_not_escape_and_names_a_file(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="dockerfile"):
            _spec("x", dockerfile=bad)


class TestSourcePaths:
    @pytest.mark.parametrize("good", ["fs-a", "other-ws/fs-a"])
    def test_a_fileset_is_a_name_or_a_workspace_qualified_name(self, good: str) -> None:
        assert FileSetSource(fileset=good).fileset == good

    @pytest.mark.parametrize("bad", ["../x", "/etc", "a/b/c", "Fs", "fs--a", ""])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="fileset"):
            FileSetSource(fileset=bad)

    @pytest.mark.parametrize("bad", ["../fs-b", "/etc", "a/../../b", "."])
    def test_a_context_path_stays_inside_the_fileset(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="context_path"):
            FileSetSource(fileset="fs-a", context_path=bad)


class TestUnknownFieldsAreRefused:
    def test_on_the_spec(self) -> None:
        body = _spec("x").model_dump() | {"dockerFile": "build/Dockerfile"}
        with pytest.raises(ValidationError, match="dockerFile"):
            BuildSpec.model_validate(body)

    def test_on_the_source(self) -> None:
        with pytest.raises(ValidationError, match="contextPath"):
            FileSetSource.model_validate({"fileset": "fs-a", "contextPath": "env"})

    def test_on_the_set(self) -> None:
        body = {"name": "demo", "revision": 1, "build_specs": [_spec("a").model_dump()], "registry": "x"}
        with pytest.raises(ValidationError, match="registry"):
            BuildSet.model_validate(body)


class TestBuildOutput:
    def test_a_registry_is_refused_rather_than_ignored(self) -> None:
        with pytest.raises(ValidationError, match="registry"):
            BuildOutput.model_validate({"registry": "elsewhere.example.com", "repository": "team/app", "tag": "v1"})

    @pytest.mark.parametrize(
        "bad", ["../team-b/app", "team/../../app", "/team/app", "team//app", "team/app/", "Team/App"]
    )
    def test_a_repository_cannot_leave_the_workspaces_path(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="repository"):
            BuildOutput(repository=bad, tag="v1")

    @pytest.mark.parametrize("bad", ["", "-v1", "v1@sha256", "a" * 129, "café"])
    def test_a_tag_must_be_an_oci_tag(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="tag"):
            BuildOutput(repository="team/app", tag=bad)

    @pytest.mark.parametrize("reserved", ["default--demo-2-0", "v1--rc", "sha256-" + "a" * 64 + ".sig"])
    def test_the_shapes_of_system_and_signature_tags_are_reserved(self, reserved: str) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            BuildOutput(repository="team/app", tag=reserved)


class TestSpecNames:
    @pytest.mark.parametrize("good", ["a", "app", "event-feed", "event_feed", "v1.2", "9lives"])
    def test_letters_and_digits_joined_by_single_separators(self, good: str) -> None:
        assert BuildSpec(name=good, source=FileSetSource(fileset="fs-a")).name == good

    def test_a_name_is_optional(self) -> None:
        assert BuildSpec(source=FileSetSource(fileset="fs-a")).name is None

    @pytest.mark.parametrize("bad", ["", "App", "my app", "a/b", "-a", "a-", "a..b", "a--b", "a__b", "café"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="name"):
            BuildSpec(name=bad, source=FileSetSource(fileset="fs-a"))


class TestOutputIsOptional:
    def test_a_spec_may_name_no_output(self) -> None:
        spec = BuildSpec(name="app", source=FileSetSource(fileset="fs-a"))
        assert spec.output is None

    def test_a_named_output_is_still_checked(self) -> None:
        with pytest.raises(ValidationError, match="repository"):
            BuildSpec.model_validate(
                {"name": "app", "source": {"fileset": "fs-a"}, "output": {"repository": "../x", "tag": "v1"}}
            )


class TestBuildSet:
    def test_revision_is_required_and_one_based(self) -> None:
        with pytest.raises(ValidationError, match="revision"):
            BuildSet(name="demo", revision=0, build_specs=[_spec("a")])
        with pytest.raises(ValidationError, match="revision"):
            BuildSet.model_validate({"name": "demo", "build_specs": [_spec("a").model_dump()]})

    def test_spec_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            BuildSet(name="demo", revision=1, build_specs=[_spec("a"), _spec("a")])

    def test_one_spec_may_leave_out_its_name(self) -> None:
        unnamed = BuildSpec(source=FileSetSource(fileset="fs-a"))
        assert BuildSet(name="demo", revision=1, build_specs=[unnamed, _spec("tests")])
        with pytest.raises(ValidationError, match="at most one"):
            BuildSet(name="demo", revision=1, build_specs=[unnamed, unnamed])

    def test_a_set_needs_at_least_one_spec(self) -> None:
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=[])

    def test_a_set_has_at_most_max_build_specs(self) -> None:
        specs = [_spec(f"s{i}") for i in range(MAX_BUILD_SPECS + 1)]
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=specs)

    @pytest.mark.parametrize(
        "bad", ["Demo", "d", "demo-", "-demo", "de--mo", "de_mo", "de.mo", "de@mo", "9demo", "demo\n"]
    )
    def test_a_name_must_fit_every_name_it_becomes(self, bad: str) -> None:
        """Row, job, tag and pod names each refuse something NAME_PATTERN allows."""
        with pytest.raises(ValidationError, match="name"):
            BuildSet(name=bad, revision=1, build_specs=[_spec("a")])

    def test_a_name_too_long_for_its_derived_names_is_refused_before_anything_is_written(self) -> None:
        # `job-fileset-<name>-<rev>`, the name Jobs gives the job's fileset: 12 + 50 + 2 = 64, one over.
        unnamed = BuildSpec(source=FileSetSource(fileset="fs-a"))
        with pytest.raises(ValidationError, match="too long"):
            BuildSet(name="a" * 50, revision=1, build_specs=[unnamed])
        assert BuildSet(name="a" * 49, revision=1, build_specs=[unnamed])

    def test_a_spec_name_too_long_for_its_image_name_is_refused(self) -> None:
        # `demo-1.<spec>`: 7 + 57 = 64, one over.
        with pytest.raises(ValidationError, match="too long"):
            BuildSet(name="demo", revision=1, build_specs=[_spec("s" * 57)])
        assert BuildSet(name="demo", revision=1, build_specs=[_spec("s" * 56)])


def _provenance(**overrides: object) -> Provenance:
    fields: dict[str, object] = {
        "build_set": "s",
        "revision": 1,
        "spec": "main",
        "job": "default/s-1",
        "request_digest": "sha256:" + "d" * 64,
    }
    return Provenance.model_validate(fields | overrides)


class TestContainerImage:
    def _pending(self) -> ContainerImage:
        return ContainerImage(
            name="s-1-0",
            workspace="default",
            registry="reg.example.com",
            repository="team/main",
            provenance=_provenance(),
        )

    def test_is_created_pending_with_nothing_observed(self) -> None:
        row = self._pending()
        assert row.status == "pending"
        assert row.digest is None and row.image_ref is None

    def test_image_ref_is_a_projection_of_the_digest(self) -> None:
        row = self._pending()
        row.digest = _DIGEST
        assert row.image_ref == f"reg.example.com/team/main@{_DIGEST}"

    def test_a_digest_must_be_sha256(self) -> None:
        with pytest.raises(ValidationError):
            ContainerImage(
                name="s-1-0",
                workspace="default",
                registry="r.example.com",
                repository="team/main",
                digest="not-a-digest",
                provenance=_provenance(),
            )

    def test_a_row_always_names_its_request(self) -> None:
        with pytest.raises(ValidationError, match="request_digest"):
            Provenance(build_set="s", revision=1, job="default/s-1")  # ty: ignore[missing-argument]

    def test_the_unnamed_spec_has_no_spec_name(self) -> None:
        assert _provenance(spec=None).spec is None
