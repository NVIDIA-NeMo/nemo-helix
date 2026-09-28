# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema rules, and the entity's write-once shape."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Provenance, Signature
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
    """Caller-supplied paths resolved inside the sandbox, rejected at submit."""

    @pytest.mark.parametrize("good", ["Dockerfile", "event_feed/Dockerfile", "a/b/c.dockerfile"])
    def test_descends(self, good: str) -> None:
        assert _spec("x", dockerfile=good).dockerfile == good

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../Dockerfile", "a/../../b/Dockerfile", "", "."])
    def test_does_not_escape_and_names_a_file(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="dockerfile"):
            _spec("x", dockerfile=bad)


class TestSourcePaths:
    """Checked at submit, where a bad value is the caller's 4xx rather than a failed fetch pod."""

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
    """A misspelt field, silently dropped, builds something other than what was asked for."""

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
    """The compiler composes the path under the workspace, so the caller's part must stay a part."""

    def test_a_registry_is_refused_rather_than_ignored(self) -> None:
        """Ignored, it would publish somewhere other than where the caller asked."""
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
        """Pushable by a caller, either would let one image stand in for another's identity."""
        with pytest.raises(ValidationError, match="reserved"):
            BuildOutput(repository="team/app", tag=reserved)


class TestBuildSet:
    def test_revision_is_required_and_one_based(self) -> None:
        with pytest.raises(ValidationError, match="revision"):
            BuildSet(name="demo", revision=0, build_specs=[_spec("a")])
        with pytest.raises(ValidationError, match="revision"):
            BuildSet.model_validate({"name": "demo", "build_specs": [_spec("a").model_dump()]})

    def test_spec_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            BuildSet(name="demo", revision=1, build_specs=[_spec("a"), _spec("a")])

    def test_a_set_needs_at_least_one_spec(self) -> None:
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=[])

    def test_a_set_has_at_most_max_build_specs(self) -> None:
        specs = [_spec(f"s{i}") for i in range(MAX_BUILD_SPECS + 1)]
        with pytest.raises(ValidationError, match="build_specs"):
            BuildSet(name="demo", revision=1, build_specs=specs)

    @pytest.mark.parametrize("bad", ["Demo", "d", "demo-", "-demo", "de--mo", "de_mo", "de.mo", "de@mo", "9demo"])
    def test_a_name_must_fit_every_name_it_becomes(self, bad: str) -> None:
        """Row, job, tag and pod names each forbid something NAME_PATTERN allows."""
        with pytest.raises(ValidationError, match="name"):
            BuildSet(name=bad, revision=1, build_specs=[_spec("a")])

    def test_a_name_too_long_for_its_derived_names_is_refused_before_anything_is_written(self) -> None:
        # `job-fileset-<name>-<rev>`, the name Jobs gives the job's fileset, is the longest name a
        # set composes while it has at most MAX_BUILD_SPECS specs: 12 + 50 + 2 = 64, one over.
        with pytest.raises(ValidationError, match="too long"):
            BuildSet(name="a" * 50, revision=1, build_specs=[_spec("a")])
        assert BuildSet(name="a" * 49, revision=1, build_specs=[_spec("a")])


class TestContainerImage:
    def _pending(self) -> ContainerImage:
        return ContainerImage(
            name="s-1-0",
            workspace="default",
            registry="reg.example.com",
            repository="team/main",
            provenance=Provenance(
                backend="execution",
                built_by=JobOrigin(build_set="s", revision=1, job="default/s-1", system_tag="default--s-1"),
            ),
        )

    def test_is_created_pending_with_nothing_observed(self) -> None:
        """The inversion: the row exists before the image does, so there is no orphan to detect."""
        row = self._pending()
        assert row.status == "pending"
        assert row.digest is None and row.manifest_digest is None and row.tag is None
        assert row.signature is None
        assert row.image_ref is None

    def test_image_ref_is_a_projection_of_the_observed_digest(self) -> None:
        row = self._pending()
        row.digest = _DIGEST
        assert row.image_ref == f"reg.example.com/team/main@{_DIGEST}"

    def test_digest_fields_reject_a_non_sha256_value(self) -> None:
        with pytest.raises(ValidationError):
            ContainerImage(
                name="s-1-0",
                workspace="default",
                registry="r.example.com",
                repository="team/main",
                digest="not-a-digest",
                provenance=Provenance(
                    backend="execution",
                    built_by=JobOrigin(build_set="s", revision=1, job="default/s-1", system_tag="default--s-1"),
                ),
            )

    def test_signature_records_presence_and_says_it_checked_nothing_more(self) -> None:
        """`verified_against` stays None in v1 and the schema is where that is stated.

        A field that overstates what was checked is worse than an absent one: anything able to
        write to the repository can place a well-formed signature made with any key.
        """
        assert Signature(storage="tag").verified_against is None

    def test_provenance_discriminates_on_origin_type(self) -> None:
        row = self._pending()
        assert isinstance(row.provenance.built_by, JobOrigin)
        assert row.provenance.built_by.system_tag == "default--s-1"
