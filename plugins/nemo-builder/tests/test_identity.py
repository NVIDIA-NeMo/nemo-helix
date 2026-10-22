# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, tags, and the system tag."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.identity import (
    ImageIdentityError,
    Platform,
    compose_system_tag,
    normalize_reference,
    validate_caller_tag,
    validate_registry_host,
)


class TestSystemTag:
    def test_is_the_workspace_and_the_image(self) -> None:
        assert compose_system_tag("scaled-evals", "freight-dispatch-shift-1.env") == (
            "scaled-evals--freight-dispatch-shift-1.env"
        )

    def test_each_image_gets_its_own_tag(self) -> None:
        assert compose_system_tag("ws", "app-1.env") != compose_system_tag("ws", "app-1.tests")

    def test_the_double_dash_keeps_workspace_and_image_apart(self) -> None:
        assert compose_system_tag("scaled-evals", "freight-1") != compose_system_tag("scaled", "evals-freight-1")

    def test_rejects_names_that_are_legal_entities_but_illegal_tags(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws+1", "app-1")
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws@1", "app-1")

    def test_rejects_a_name_already_containing_the_separator(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("we--ird", "app-1")


class TestCallerTags:
    def test_an_ordinary_tag_is_allowed(self) -> None:
        assert validate_caller_tag("v1.2-rc_3") == "v1.2-rc_3"

    @pytest.mark.parametrize(
        "reserved", [compose_system_tag("default", "demo-2.main"), "a--b", "sha256-" + "a" * 64 + ".sig"]
    )
    def test_system_and_signature_shapes_are_reserved(self, reserved: str) -> None:
        with pytest.raises(ImageIdentityError, match="reserved"):
            validate_caller_tag(reserved)

    @pytest.mark.parametrize("bad", ["café", "v１", "日本"])
    def test_a_tag_is_ascii(self, bad: str) -> None:
        with pytest.raises(ImageIdentityError):
            validate_caller_tag(bad)


class TestRegistryHost:
    @pytest.mark.parametrize("good", ["registry.example.com", "localhost:5000", "Registry.Example.com"])
    def test_a_host_is_lowercased(self, good: str) -> None:
        assert validate_registry_host(good) == good.lower()

    @pytest.mark.parametrize("bad", ["", "https:", "user@host", "host name", "reg?x=1", "host:port"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ImageIdentityError):
            validate_registry_host(bad)


class TestNormalizeReference:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("alpine", "docker.io/library/alpine:latest"),
            ("ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
            ("org/app", "docker.io/org/app:latest"),
            ("index.docker.io/library/python:3.12", "docker.io/library/python:3.12"),
            ("ghcr.io/org/app:1", "ghcr.io/org/app:1"),
            ("localhost:5000/team/app", "localhost:5000/team/app:latest"),
            ("redis:7@sha256:" + "a" * 64, "docker.io/library/redis@sha256:" + "a" * 64),
        ],
    )
    def test_docker_shorthand_is_expanded(self, given: str, expected: str) -> None:
        """Two spellings of one image must normalize alike, or the allowlist is decided by spelling."""
        assert str(normalize_reference(given)) == expected

    @pytest.mark.parametrize("bad", ["", "a b", "https://ghcr.io/x", "ubuntu@sha256:abc", "Ubuntu"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ImageIdentityError):
            normalize_reference(bad)


class TestPlatform:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("linux/amd64", "linux/amd64"),
            ("linux/x86_64", "linux/amd64"),
            ("Linux/X86-64", "linux/amd64"),
            ("linux/aarch64", "linux/arm64"),
            ("linux/arm64/v8", "linux/arm64"),
            ("linux/arm/v7", "linux/arm/v7"),
        ],
    )
    def test_is_named_as_buildkit_names_it(self, value: str, expected: str) -> None:
        assert str(Platform.parse(value)) == expected

    @pytest.mark.parametrize("value", ["", "arm64", "linux/", "linux/arm64/v8/x", "linux/$TARGETARCH"])
    def test_refuses_anything_but_os_and_architecture(self, value: str) -> None:
        with pytest.raises(ImageIdentityError):
            Platform.parse(value)

    def test_reads_an_oci_platform(self) -> None:
        assert Platform.from_oci({"os": "linux", "architecture": "arm64", "variant": "v8"}) == Platform.parse(
            "linux/arm64"
        )
