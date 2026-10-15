# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import agents as agents_router_module
from nemo_agents_plugin.api.v2.dependencies import get_registry_http_client
from nemo_agents_plugin.container.image_bundle import (
    DOCKER_HUB_REGISTRY,
    ImageAccessDeniedError,
    ImageBundleError,
    ImageNotFoundError,
    ImageReference,
    parse_image_reference,
    read_agent_bundle,
)

AGENT_YAML = b"config_format: nemo-agents-spec-v1\nname: triage\n"


def _layer(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue())


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class FakeRegistry:
    """A registry serving one packaged image, behind Docker Hub style anonymous token auth."""

    def __init__(
        self,
        *,
        bundle: dict[str, bytes] | None = None,
        env: list[str] | None = None,
        copy_command: str = "COPY ./ /workspace # buildkit",
        indexed: bool = True,
        private: bool = False,
    ) -> None:
        bundle = (
            {"workspace/agent.yaml": AGENT_YAML, "workspace/tools/search.py": b"print(1)\n"}
            if bundle is None
            else bundle
        )
        self.layers = [_layer({"etc/os-release": b"ubuntu"}), _layer(bundle), _layer({"workspace/.venv/x": b"x"})]
        self.config = json.dumps(
            {
                "config": {"Env": env if env is not None else ["PATH=/bin", "AGENT_CONFIG_PATH=/workspace/agent.yaml"]},
                "history": [
                    {"created_by": "/bin/sh -c #(nop) ADD file:abc in /"},
                    {"created_by": "ENV PORT=8000", "empty_layer": True},
                    {"created_by": copy_command},
                    {"created_by": "RUN uv venv /workspace/.venv"},
                ],
            }
        ).encode()
        self.manifest = json.dumps(
            {
                "config": {"digest": _digest(self.config)},
                "layers": [{"digest": _digest(layer), "size": len(layer)} for layer in self.layers],
            }
        ).encode()
        self.indexed = indexed
        self.private = private
        self.blob_requests: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "anon"})
        if request.headers.get("authorization") != "Bearer anon" or self.private:
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer realm="https://auth.docker.io/token",service="registry.docker.io"'
                },
            )
        path = request.url.path
        if "/manifests/" in path:
            ref = path.rsplit("/", 1)[1]
            if ref == "0.1" and self.indexed:
                return httpx.Response(
                    200,
                    json={
                        "manifests": [
                            {"digest": "sha256:attestation", "platform": {"os": "unknown", "architecture": "unknown"}},
                            {"digest": _digest(self.manifest), "platform": {"os": "linux", "architecture": "amd64"}},
                        ]
                    },
                )
            if ref in ("0.1", _digest(self.manifest)):
                return httpx.Response(200, content=self.manifest)
            return httpx.Response(404)
        digest = path.rsplit("/", 1)[1]
        self.blob_requests.append(digest)
        if digest == _digest(self.config):
            return httpx.Response(200, content=self.config)
        for layer in self.layers:
            if digest == _digest(layer):
                return httpx.Response(200, content=layer)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


IMAGE = "docker.io/sgavrilovic951/sample_agent:0.1"


class TestParseImageReference:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (IMAGE, ImageReference(DOCKER_HUB_REGISTRY, "sgavrilovic951/sample_agent", "0.1")),
            ("ubuntu", ImageReference(DOCKER_HUB_REGISTRY, "library/ubuntu", "latest")),
            ("nvcr.io/nvidia/nemo/agent:1.0", ImageReference("nvcr.io", "nvidia/nemo/agent", "1.0")),
            (
                "registry.example.com:5000/team/agent",
                ImageReference("registry.example.com:5000", "team/agent", "latest"),
            ),
            (f"org/agent@sha256:{'a' * 64}", ImageReference(DOCKER_HUB_REGISTRY, "org/agent", f"sha256:{'a' * 64}")),
        ],
    )
    def test_resolves_registry_repository_and_reference(self, image: str, expected: ImageReference) -> None:
        assert parse_image_reference(image) == expected

    @pytest.mark.parametrize("image", ["", "https://docker.io/org/agent", "org agent"])
    def test_rejects_what_is_not_a_reference(self, image: str) -> None:
        with pytest.raises(ImageBundleError, match="not an image reference"):
            parse_image_reference(image)

    @pytest.mark.parametrize("image", ["localhost:5000/agent", "10.0.0.5:5000/agent", "169.254.169.254/agent"])
    def test_refuses_registries_on_internal_networks(self, image: str) -> None:
        with pytest.raises(ImageBundleError):
            parse_image_reference(image)


class TestReadAgentBundle:
    async def test_returns_the_agent_directory_from_the_copy_layer_only(self) -> None:
        registry = FakeRegistry()
        async with registry.client() as http:
            bundle = await read_agent_bundle(IMAGE, http=http)

        assert bundle.config_path == "/workspace/agent.yaml"
        assert bundle.files == {"agent.yaml": AGENT_YAML, "tools/search.py": b"print(1)\n"}
        assert _digest(registry.layers[2]) not in registry.blob_requests
        assert _digest(registry.layers[0]) not in registry.blob_requests

    async def test_reads_a_single_platform_manifest(self) -> None:
        async with FakeRegistry(indexed=False).client() as http:
            bundle = await read_agent_bundle(IMAGE, http=http)

        assert "agent.yaml" in bundle.files

    async def test_leaves_out_build_artifacts_and_whiteouts(self) -> None:
        registry = FakeRegistry(
            bundle={
                "workspace/agent.yaml": AGENT_YAML,
                "workspace/Dockerfile.generated": b"FROM x",
                "workspace/.dockerignore": b".env",
                "workspace/Dockerfile.generated.dockerignore": b".env",
                "workspace/nemo_helix-1.0-py3-none-any.whl": b"\x00",
                "workspace/old/.wh.gone.py": b"",
            }
        )
        async with registry.client() as http:
            bundle = await read_agent_bundle(IMAGE, http=http)

        assert list(bundle.files) == ["agent.yaml"]

    async def test_names_a_differently_named_config_agent_yaml(self) -> None:
        registry = FakeRegistry(
            bundle={"app/triage.yaml": AGENT_YAML},
            env=["AGENT_CONFIG_PATH=/app/triage.yaml"],
            copy_command="COPY ./ /app",
        )
        async with registry.client() as http:
            bundle = await read_agent_bundle(IMAGE, http=http)

        assert bundle.files == {"agent.yaml": AGENT_YAML}

    async def test_requires_agent_config_path(self) -> None:
        async with FakeRegistry(env=["PATH=/bin"]).client() as http:
            with pytest.raises(ImageBundleError, match="AGENT_CONFIG_PATH"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_requires_the_copy_layer(self) -> None:
        async with FakeRegistry(copy_command="RUN true").client() as http:
            with pytest.raises(ImageBundleError, match="No layer copies the agent"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_requires_agent_yaml_in_the_layer(self) -> None:
        async with FakeRegistry(bundle={"workspace/README.md": b"hi"}).client() as http:
            with pytest.raises(ImageBundleError, match="has no agent.yaml"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_reports_a_missing_image(self) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageNotFoundError):
                await read_agent_bundle("docker.io/sgavrilovic951/sample_agent:9.9", http=http)

    async def test_reports_a_private_image(self) -> None:
        async with FakeRegistry(private=True).client() as http:
            with pytest.raises(ImageAccessDeniedError, match="private"):
                await read_agent_bundle(IMAGE, http=http)


def _api_client(registry: FakeRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(agents_router_module.router, prefix="/apis/agents/v2/workspaces/{workspace}")

    async def http() -> Any:
        async with registry.client() as client:
            yield client

    app.dependency_overrides[get_registry_http_client] = http
    return TestClient(app, raise_server_exceptions=False)


class TestReadAgentImageBundleRoute:
    URL = "/apis/agents/v2/workspaces/default/agent-images/bundle"

    def test_returns_base64_files(self) -> None:
        response = _api_client(FakeRegistry()).post(self.URL, json={"image": f" {IMAGE} "})

        assert response.status_code == 200
        body = response.json()
        assert body["image"] == IMAGE
        assert body["config_path"] == "/workspace/agent.yaml"
        assert {item["path"]: base64.b64decode(item["content"]) for item in body["files"]} == {
            "agent.yaml": AGENT_YAML,
            "tools/search.py": b"print(1)\n",
        }

    @pytest.mark.parametrize(
        ("registry", "image", "status"),
        [
            (FakeRegistry(), "docker.io/sgavrilovic951/sample_agent:9.9", 404),
            (FakeRegistry(private=True), IMAGE, 403),
            (FakeRegistry(env=[]), IMAGE, 422),
            (FakeRegistry(), "https://hub.docker.com/r/org/agent", 422),
        ],
    )
    def test_maps_failures_to_status_codes(self, registry: FakeRegistry, image: str, status: int) -> None:
        response = _api_client(registry).post(self.URL, json={"image": image})

        assert response.status_code == status
        assert response.json()["detail"]
