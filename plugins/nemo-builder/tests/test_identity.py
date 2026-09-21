# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity rules that a bypass would hide behind."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.identity import (
    ImageIdentityError,
    compose_system_tag,
    normalize_digest,
    normalize_reference,
    parse_reference,
    split_system_tag,
)

_DIGEST = "sha256:" + "a" * 64


class TestNormalization:
    """The five spellings of one image.

    This is the test RFC 001 asks for by name, and it is the one that matters: normalization and
    any future allowlist are one feature, because a raw string comparison against a list holding
    `docker.io` matches none of the first three spellings below. A caller who can pick the
    spelling would otherwise pick the verdict.
    """

    @pytest.mark.parametrize(
        "spelling",
        [
            "alpine",
            "alpine:latest",
            "library/alpine:latest",
            "docker.io/library/alpine:latest",
        ],
    )
    def test_all_spellings_of_one_image_agree(self, spelling: str) -> None:
        assert normalize_reference(spelling).normalized_ref == "docker.io/library/alpine:latest"

    def test_multi_component_repository_gets_no_implicit_library(self) -> None:
        assert normalize_reference("bitnami/nginx").normalized_ref == "docker.io/bitnami/nginx:latest"

    def test_an_explicit_registry_is_left_alone(self) -> None:
        ref = normalize_reference("us-central1-docker.pkg.dev/proj/repo/img:v1")
        assert ref.registry == "us-central1-docker.pkg.dev"
        assert ref.repository == "proj/repo/img"
        assert ref.tag == "v1"

    def test_a_host_is_told_from_a_repository_by_dot_colon_or_localhost(self) -> None:
        # `alpine/foo` is a Docker Hub repository, NOT a registry named `alpine`.
        assert normalize_reference("alpine/foo").registry == "docker.io"
        assert normalize_reference("localhost:5000/foo:v1").registry == "localhost:5000"
        assert normalize_reference("reg.example.com/foo:v1").registry == "reg.example.com"


class TestParsing:
    def test_digest_reference_round_trips(self) -> None:
        ref = parse_reference(f"reg.example.com/team/app@{_DIGEST}")
        assert ref.digest == _DIGEST
        assert ref.tag is None
        assert ref.normalized_ref == f"reg.example.com/team/app@{_DIGEST}"

    def test_image_ref_projection_is_registry_repository_at_digest(self) -> None:
        ref = parse_reference("reg.example.com/team/app:v1")
        assert ref.digest_ref(_DIGEST) == f"reg.example.com/team/app@{_DIGEST}"

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "https://reg.example.com/app:v1",  # a URL is not a reference
            "reg.example.com/app",  # neither tag nor digest
            f"reg.example.com/app:v1@{_DIGEST}",  # tag-plus-digest
            f"reg.example.com/app@{_DIGEST}@{_DIGEST}",
            "reg.example.com/App:v1",  # uppercase repository component
            "app:v1",  # no explicit registry -- parse_reference does not expand
        ],
    )
    def test_rejects(self, bad: str) -> None:
        with pytest.raises(ImageIdentityError):
            parse_reference(bad)

    def test_digest_must_be_lowercase_hex(self) -> None:
        assert normalize_digest("SHA256:" + "A" * 64) == _DIGEST
        with pytest.raises(ImageIdentityError):
            normalize_digest("sha256:" + "g" * 64)
        with pytest.raises(ImageIdentityError):
            normalize_digest("sha512:" + "a" * 64)


class TestSystemTag:
    def test_composes_and_splits(self) -> None:
        tag = compose_system_tag("scaled-evals", "freight-dispatch-shift", 1)
        assert tag == "scaled-evals--freight-dispatch-shift-1"
        assert split_system_tag(tag) == ("scaled-evals", "freight-dispatch-shift", 1)

    def test_the_double_dash_is_what_makes_the_split_unambiguous(self) -> None:
        """A single `-` separator would collide these two; `--` does not.

        Both workspace and set name may contain `-`, so `scaled-evals` + `freight` and
        `scaled` + `evals-freight` compose identically under a single dash. This is the whole
        reason for the separator, so it gets a test rather than a comment.
        """
        a = compose_system_tag("scaled-evals", "freight", 1)
        b = compose_system_tag("scaled", "evals-freight", 1)
        assert a != b
        assert split_system_tag(a) == ("scaled-evals", "freight", 1)
        assert split_system_tag(b) == ("scaled", "evals-freight", 1)

    def test_rejects_names_that_are_legal_entities_but_illegal_tags(self) -> None:
        """NAME_PATTERN admits `@` and `+`; an OCI tag forbids both.

        Caught at composition, not at push twenty minutes later.
        """
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "build+set", 1)
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "build@set", 1)

    def test_rejects_a_name_already_containing_the_separator(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("we--ird", "set", 1)

    def test_revision_is_one_based(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "set", 0)
