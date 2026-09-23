# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema rules, and the entity's write-once shape."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Provenance, Signature
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from pydantic import ValidationError

_DIGEST = "sha256:" + "b" * 64


def _spec(
    name: str, *, fileset: str = "fs-a", context_path: str | None = None, dockerfile: str = "Dockerfile"
) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(registry="reg.example.com", repository=f"team/{name}", tag="v1"),
        dockerfile=dockerfile,
    )


class TestDockerfileStaysWithinContext:
    """Caller-supplied paths resolved inside the sandbox, rejected at submit."""

    @pytest.mark.parametrize("good", ["Dockerfile", "event_feed/Dockerfile", "a/b/c.dockerfile"])
    def test_descends(self, good: str) -> None:
        assert _spec("x", dockerfile=good).dockerfile == good

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../Dockerfile", "a/../../b/Dockerfile"])
    def test_does_not_escape(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _spec("x", dockerfile=bad)


class TestBuildSet:
    def test_revision_is_required_and_one_based(self) -> None:
        with pytest.raises(ValidationError):
            BuildSet(name="s", revision=0, build_specs=[_spec("a")])
        with pytest.raises(ValidationError):
            BuildSet.model_validate({"name": "s", "build_specs": [_spec("a").model_dump()]})

    def test_spec_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            BuildSet(name="s", revision=1, build_specs=[_spec("a"), _spec("a")])

    def test_a_set_needs_at_least_one_spec(self) -> None:
        with pytest.raises(ValidationError):
            BuildSet(name="s", revision=1, build_specs=[])


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
