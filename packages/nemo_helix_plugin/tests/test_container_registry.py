# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix_plugin.container_registry import (
    DOCKER_HUB_REGISTRY,
    ImageReference,
    ImageReferenceError,
    normalize_registry,
    parse_challenges,
    parse_image_reference,
)

DIGEST = f"sha256:{'a' * 64}"


class TestParseImageReference:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            ("docker.io/org/sample_agent:0.1", ImageReference(DOCKER_HUB_REGISTRY, "org/sample_agent", "0.1")),
            ("ubuntu", ImageReference(DOCKER_HUB_REGISTRY, "library/ubuntu", "latest")),
            ("docker.io/ubuntu", ImageReference(DOCKER_HUB_REGISTRY, "library/ubuntu", "latest")),
            ("nvcr.io/nvidia/nemo/agent:1.0", ImageReference("nvcr.io", "nvidia/nemo/agent", "1.0")),
            (
                "Registry.Example.com:5000/team/agent",
                ImageReference("registry.example.com:5000", "team/agent", "latest"),
            ),
            ("ghcr.io:443/org/agent", ImageReference("ghcr.io", "org/agent", "latest")),
            ("localhost:5000/agent", ImageReference("localhost:5000", "agent", "latest")),
            ("[::1]:5000/agent", ImageReference("[::1]:5000", "agent", "latest")),
            (f"org/agent@{DIGEST}", ImageReference(DOCKER_HUB_REGISTRY, "org/agent", DIGEST)),
            (f"org/agent:1.0@{DIGEST}", ImageReference(DOCKER_HUB_REGISTRY, "org/agent", DIGEST)),
        ],
    )
    def test_resolves_registry_repository_and_reference(self, image: str, expected: ImageReference) -> None:
        assert parse_image_reference(image) == expected

    @pytest.mark.parametrize(
        "image",
        [
            "",
            "https://docker.io/org/agent",
            "org agent",
            "docker.io/",
            "org//agent",
            "Org/Agent",
            "org/agent:",
            "org/agent@",
            "org/agent@sha256:../../x",
            "org/agent:../x",
            "bad_host!/org/agent",
        ],
    )
    def test_rejects_what_is_not_a_reference(self, image: str) -> None:
        with pytest.raises(ImageReferenceError, match="not an image reference"):
            parse_image_reference(image)

    def test_rejects_an_overlong_reference_before_matching_it(self) -> None:
        with pytest.raises(ImageReferenceError, match="at most 512 characters"):
            parse_image_reference("0" * 100_000)

    @pytest.mark.parametrize(
        "image",
        [
            f"{'0' * 400}!/agent",
            f"org/agent@0{'+0' * 200}",
            f"org/agent@{'0+' * 200}",
            f"org/{'a-' * 200}!",
        ],
    )
    def test_rejects_crafted_references_at_the_length_limit(self, image: str) -> None:
        with pytest.raises(ImageReferenceError):
            parse_image_reference(image)

    @pytest.mark.parametrize(
        ("image", "printed"),
        [
            ("org/agent", "registry-1.docker.io/org/agent:latest"),
            (f"org/agent@{DIGEST}", f"registry-1.docker.io/org/agent@{DIGEST}"),
        ],
    )
    def test_prints_a_valid_reference(self, image: str, printed: str) -> None:
        assert str(parse_image_reference(image)) == printed


@pytest.mark.parametrize(
    ("registry", "normalized"),
    [
        ("docker.io", DOCKER_HUB_REGISTRY),
        (" Index.Docker.io ", DOCKER_HUB_REGISTRY),
        ("ghcr.io:443", "ghcr.io"),
        ("harbor.corp:5000", "harbor.corp:5000"),
    ],
)
def test_normalize_registry(registry: str, normalized: str) -> None:
    assert normalize_registry(registry) == normalized


class TestParseChallenges:
    def test_reads_several_challenges_and_quoted_values(self) -> None:
        header = 'Bearer realm="https://auth.example/token",service="registry", Basic realm="x"'
        assert dict(parse_challenges(header)) == {
            "bearer": {"realm": "https://auth.example/token", "service": "registry"},
            "basic": {"realm": "x"},
        }

    def test_unescapes_quoted_pairs(self) -> None:
        assert parse_challenges('Bearer realm="a\\"b"') == [("bearer", {"realm": 'a"b'})]
