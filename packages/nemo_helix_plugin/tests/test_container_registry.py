# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix_plugin.container_registry import (
    DOCKER_HUB_REGISTRY,
    ImageReference,
    ImageReferenceError,
    parse_challenges,
    parse_image_reference,
)


class TestParseImageReference:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            ("docker.io/org/sample_agent:0.1", ImageReference(DOCKER_HUB_REGISTRY, "org/sample_agent", "0.1")),
            ("ubuntu", ImageReference(DOCKER_HUB_REGISTRY, "library/ubuntu", "latest")),
            ("nvcr.io/nvidia/nemo/agent:1.0", ImageReference("nvcr.io", "nvidia/nemo/agent", "1.0")),
            (
                "Registry.Example.com:5000/team/agent",
                ImageReference("registry.example.com:5000", "team/agent", "latest"),
            ),
            ("localhost:5000/agent", ImageReference("localhost:5000", "agent", "latest")),
            ("[::1]:5000/agent", ImageReference("[::1]:5000", "agent", "latest")),
            (f"org/agent@sha256:{'a' * 64}", ImageReference(DOCKER_HUB_REGISTRY, "org/agent", f"sha256:{'a' * 64}")),
        ],
    )
    def test_resolves_registry_repository_and_reference(self, image: str, expected: ImageReference) -> None:
        assert parse_image_reference(image) == expected

    @pytest.mark.parametrize("image", ["", "https://docker.io/org/agent", "org agent", "docker.io/"])
    def test_rejects_what_is_not_a_reference(self, image: str) -> None:
        with pytest.raises(ImageReferenceError, match="not an image reference"):
            parse_image_reference(image)

    def test_prints_like_docker(self) -> None:
        assert str(parse_image_reference("org/agent")) == "registry-1.docker.io/org/agent:latest"


class TestParseChallenges:
    def test_reads_several_challenges_and_quoted_values(self) -> None:
        header = 'Bearer realm="https://auth.example/token",service="registry", Basic realm="x"'
        assert dict(parse_challenges(header)) == {
            "bearer": {"realm": "https://auth.example/token", "service": "registry"},
            "basic": {"realm": "x"},
        }

    def test_unescapes_quoted_pairs(self) -> None:
        assert parse_challenges('Bearer realm="a\\"b"') == [("bearer", {"realm": 'a"b'})]
