# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving an image to copy, against a fake registry."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
from nemo_builder_plugin.identity import normalize_reference
from nemo_builder_plugin.upstream import UpstreamError, UpstreamNotFound, UpstreamRegistry

INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _registry(documents: dict[str, tuple[bytes, str]], *, token: bool = False) -> UpstreamRegistry:
    """A registry serving ``documents`` by path; with ``token``, only to a bearer token from its realm."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "auth.example.com":
            assert request.url.params["scope"] == "repository:library/ubuntu:pull"
            return httpx.Response(200, json={"token": "t0k"})
        if token and request.headers.get("Authorization") != "Bearer t0k":
            challenge = 'Bearer realm="https://auth.example.com/token",service="registry.docker.io"'
            return httpx.Response(401, headers={"WWW-Authenticate": challenge})
        found = documents.get(request.url.path)
        if found is None:
            return httpx.Response(404)
        content, media_type = found
        return httpx.Response(200, content=content, headers={"Content-Type": media_type})

    return UpstreamRegistry(transport=httpx.MockTransport(handler))


def _single(architecture: str) -> tuple[bytes, bytes]:
    config = json.dumps({"os": "linux", "architecture": architecture}).encode()
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": _digest(config)}, "layers": []}).encode()
    return manifest, config


BASE = "/v2/library/ubuntu"


def test_an_index_resolves_to_its_child_for_the_platform() -> None:
    amd64, arm64 = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    index = json.dumps(
        {
            "manifests": [
                {"digest": amd64, "platform": {"os": "linux", "architecture": "amd64"}},
                {"digest": arm64, "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"}},
            ]
        }
    ).encode()
    registry = _registry({f"{BASE}/manifests/24.04": (index, INDEX)}, token=True)

    resolved = registry.resolve(normalize_reference("ubuntu:24.04"), platform="linux/arm64")

    assert resolved.manifest_digest == arm64


def test_a_single_image_resolves_to_itself_when_its_config_is_for_the_platform() -> None:
    manifest, config = _single("amd64")
    registry = _registry(
        {f"{BASE}/manifests/24.04": (manifest, MANIFEST), f"{BASE}/blobs/{_digest(config)}": (config, "")}
    )

    assert registry.resolve(normalize_reference("ubuntu:24.04"), platform="linux/amd64").manifest_digest == _digest(
        manifest
    )


def test_a_single_image_for_another_platform_is_not_found() -> None:
    manifest, config = _single("arm64")
    registry = _registry(
        {f"{BASE}/manifests/24.04": (manifest, MANIFEST), f"{BASE}/blobs/{_digest(config)}": (config, "")}
    )

    with pytest.raises(UpstreamNotFound, match="isn't for linux/amd64"):
        registry.resolve(normalize_reference("ubuntu:24.04"), platform="linux/amd64")


def test_an_index_without_the_platform_is_not_found_rather_than_its_first_entry() -> None:
    index = json.dumps(
        {"manifests": [{"digest": "sha256:" + "a" * 64, "platform": {"os": "linux", "architecture": "s390x"}}]}
    )
    registry = _registry({f"{BASE}/manifests/24.04": (index.encode(), INDEX)})

    with pytest.raises(UpstreamNotFound, match="no linux/amd64 image"):
        registry.resolve(normalize_reference("ubuntu:24.04"), platform="linux/amd64")


def test_a_missing_or_private_image_is_not_found() -> None:
    with pytest.raises(UpstreamNotFound):
        _registry({}).resolve(normalize_reference("ubuntu:24.04"), platform="linux/amd64")


def test_a_digest_reference_served_as_other_bytes_is_refused() -> None:
    manifest, config = _single("amd64")
    asked = "sha256:" + "9" * 64
    registry = _registry({f"{BASE}/manifests/{asked}": (manifest, MANIFEST)})

    with pytest.raises(UpstreamError, match="served as"):
        registry.resolve(normalize_reference(f"ubuntu@{asked}"), platform="linux/amd64")


@pytest.mark.parametrize(
    "entry",
    [
        {"platform": {"os": "linux", "architecture": "amd64"}},
        {"digest": "md5:abc", "platform": {"os": "linux", "architecture": "amd64"}},
    ],
)
def test_an_index_entry_for_the_platform_without_a_sha256_digest_is_an_upstream_error(entry: dict) -> None:
    index = json.dumps({"manifests": [{"platform": "not an object"}, entry]}).encode()
    registry = _registry({f"{BASE}/manifests/24.04": (index, INDEX)})

    with pytest.raises(UpstreamError, match="without a sha256 digest"):
        registry.resolve(normalize_reference("ubuntu:24.04"), platform="linux/amd64")


def test_an_index_lists_its_platforms_without_attestations_or_arm64s_default_variant() -> None:
    index = json.dumps(
        {
            "manifests": [
                {"digest": "sha256:" + "a" * 64, "platform": {"os": "linux", "architecture": "amd64"}},
                {"digest": "sha256:" + "b" * 64, "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"}},
                {"digest": "sha256:" + "c" * 64, "platform": {"os": "linux", "architecture": "arm", "variant": "v7"}},
                {"digest": "sha256:" + "d" * 64, "platform": {"os": "unknown", "architecture": "unknown"}},
            ]
        }
    )
    registry = _registry({f"{BASE}/manifests/24.04": (index.encode(), INDEX)})

    assert registry.platforms(normalize_reference("ubuntu:24.04")) == ["linux/amd64", "linux/arm/v7", "linux/arm64"]


def test_a_single_image_lists_its_configs_platform() -> None:
    manifest, config = _single("arm64")
    registry = _registry(
        {f"{BASE}/manifests/24.04": (manifest, MANIFEST), f"{BASE}/blobs/{_digest(config)}": (config, "")}
    )

    assert registry.platforms(normalize_reference("ubuntu:24.04")) == ["linux/arm64"]
