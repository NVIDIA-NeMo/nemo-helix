# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import ipaddress
import json
import socket
import tarfile
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import agents as agents_router_module
from nemo_agents_plugin.api.v2.dependencies import get_registry_http_client
from nemo_agents_plugin.container import image_bundle
from nemo_agents_plugin.container.image_bundle import (
    DOCKER_HUB_REGISTRY,
    IMAGE_REGISTRIES_ENV,
    ImageAccessDeniedError,
    ImageBundleError,
    ImageNotFoundError,
    ImageReference,
    parse_image_reference,
    read_agent_bundle,
)

AGENT_YAML = b"config_format: nemo-agents-spec-v1\nname: triage\n"
PUBLIC_ADDRESS = ipaddress.ip_address("93.184.216.34")
CDN_HOST = "cdn.example.net"


@pytest.fixture(autouse=True)
def resolve_everything_publicly(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    answers: dict[str, list[Any]] = {}

    def resolve(host: str) -> list[Any]:
        if host in answers:
            return answers[host]
        try:
            return [ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))]
        except OSError:
            return [PUBLIC_ADDRESS]

    monkeypatch.setattr(image_bundle, "_resolve_host", resolve)
    return answers


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
        redirect_blobs_to: str | None = None,
        manifest_body: bytes | None = None,
        layer_body: bytes | None = None,
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
        self.redirect_blobs_to = redirect_blobs_to
        self.manifest_body = manifest_body
        self.layer_body = layer_body
        self.blob_requests: list[str] = []
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "anon"})
        if request.url.host == CDN_HOST:
            return self._blob(request.url.path.rsplit("/", 1)[1])
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
                return httpx.Response(200, content=self.manifest if self.manifest_body is None else self.manifest_body)
            return httpx.Response(404)
        digest = path.rsplit("/", 1)[1]
        self.blob_requests.append(digest)
        if self.redirect_blobs_to is not None:
            return httpx.Response(307, headers={"location": f"{self.redirect_blobs_to}/blobs/{digest}"})
        return self._blob(digest)

    def _blob(self, digest: str) -> httpx.Response:
        if digest == _digest(self.config):
            return httpx.Response(200, content=self.config)
        for layer in self.layers:
            if digest == _digest(layer):
                return httpx.Response(200, content=layer if self.layer_body is None else self.layer_body)
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


class TestRegistryPolicy:
    async def test_refuses_registries_outside_the_allow_list(self) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="not an allowed image registry"):
                await read_agent_bundle("registry.example.com/team/agent", http=http)

    async def test_allow_list_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(IMAGE_REGISTRIES_ENV, "ghcr.io, Registry.Example.com")
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="not an allowed image registry"):
                await read_agent_bundle(IMAGE, http=http)
            with pytest.raises(ImageNotFoundError):
                await read_agent_bundle("registry.example.com/team/agent", http=http)

    @pytest.mark.parametrize(
        "image",
        [
            "localhost:5000/agent",
            "10.0.0.5:5000/agent",
            "169.254.169.254/agent",
            "127.1/agent",
            "[::1]:5000/agent",
            "[::ffff:127.0.0.1]/agent",
            "registry.svc:5000/agent",
            "intranet-registry:5000/agent",
            "metadata.internal/agent",
        ],
    )
    async def test_refuses_registries_on_internal_networks(self, image: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(IMAGE_REGISTRIES_ENV, "*")
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="private network"):
                await read_agent_bundle(image, http=http)

    async def test_refuses_registries_that_resolve_to_internal_addresses(
        self, monkeypatch: pytest.MonkeyPatch, resolve_everything_publicly: dict[str, list[Any]]
    ) -> None:
        monkeypatch.setenv(IMAGE_REGISTRIES_ENV, "*")
        resolve_everything_publicly["registry.example.com"] = [PUBLIC_ADDRESS, ipaddress.ip_address("10.0.0.5")]
        registry = FakeRegistry()
        async with registry.client() as http:
            with pytest.raises(ImageBundleError, match="private network"):
                await read_agent_bundle("registry.example.com/team/agent", http=http)
        assert registry.requests == []

    async def test_refuses_registries_that_do_not_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(IMAGE_REGISTRIES_ENV, "*")

        def unresolvable(host: str) -> list[Any]:
            raise OSError("no such host")

        monkeypatch.setattr(image_bundle, "_resolve_host", unresolvable)
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="could not be resolved"):
                await read_agent_bundle("registry.example.com/team/agent", http=http)

    async def test_refuses_token_services_that_are_not_public_https(self) -> None:
        registry = FakeRegistry()
        original = registry.handle

        def handle(request: httpx.Request) -> httpx.Response:
            response = original(request)
            if response.status_code == 401:
                response.headers["www-authenticate"] = 'Bearer realm="http://169.254.169.254/token"'
            return response

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            with pytest.raises(ImageBundleError, match="token service"):
                await read_agent_bundle(IMAGE, http=http)
        assert all(request.url.host != "169.254.169.254" for request in registry.requests)


class TestRedirects:
    async def test_follows_https_blob_redirects_without_forwarding_the_token(self) -> None:
        registry = FakeRegistry(redirect_blobs_to=f"https://{CDN_HOST}")
        async with registry.client() as http:
            bundle = await read_agent_bundle(IMAGE, http=http)

        assert "agent.yaml" in bundle.files
        cdn_requests = [request for request in registry.requests if request.url.host == CDN_HOST]
        assert cdn_requests
        assert all("authorization" not in request.headers for request in cdn_requests)

    @pytest.mark.parametrize("target", [f"http://{CDN_HOST}", "https://10.0.0.5", "https://169.254.169.254"])
    async def test_refuses_redirects_off_public_https(self, target: str) -> None:
        registry = FakeRegistry(redirect_blobs_to=target)
        async with registry.client() as http:
            with pytest.raises(ImageBundleError):
                await read_agent_bundle(IMAGE, http=http)
        assert all(request.url.host in (DOCKER_HUB_REGISTRY, "auth.docker.io") for request in registry.requests)

    async def test_gives_up_after_too_many_redirects(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": str(request.url)})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            with pytest.raises(ImageBundleError, match="too many times"):
                await read_agent_bundle(IMAGE, http=http)


class TestBodyLimits:
    async def test_caps_the_layer_regardless_of_the_declared_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_BUNDLE_LAYER_BYTES", 1024)
        registry = FakeRegistry(layer_body=b"\0" * 2048)
        async with registry.client() as http:
            with pytest.raises(ImageBundleError, match="agent layer is larger than"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_caps_metadata_bodies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_METADATA_BYTES", 64)
        async with FakeRegistry(manifest_body=b"{" + b" " * 128 + b"}").client() as http:
            with pytest.raises(ImageBundleError, match="manifest is larger than"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_caps_how_far_the_layer_inflates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_INFLATED_LAYER_BYTES", 1024 * 1024)
        bomb = _layer({"workspace/agent.yaml": AGENT_YAML, "workspace/pad.bin": b"\0" * (4 * 1024 * 1024)})
        async with FakeRegistry(layer_body=bomb).client() as http:
            with pytest.raises(ImageBundleError, match="inflates to more than"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_rejects_zstd_layers(self) -> None:
        async with FakeRegistry(layer_body=b"\x28\xb5\x2f\xfd" + b"\0" * 16).client() as http:
            with pytest.raises(ImageBundleError, match="zstd"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_reports_malformed_registry_answers(self) -> None:
        async with FakeRegistry(manifest_body=b"<html>busy</html>").client() as http:
            with pytest.raises(ImageBundleError, match="manifest is not JSON"):
                await read_agent_bundle(IMAGE, http=http)

    async def test_does_not_echo_registry_status_codes(self) -> None:
        registry = FakeRegistry()

        def handle(request: httpx.Request) -> httpx.Response:
            response = registry.handle(request)
            if "/manifests/" in request.url.path and response.status_code == 200:
                return httpx.Response(503)
            return response

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            with pytest.raises(ImageBundleError, match="unexpected response") as excinfo:
                await read_agent_bundle(IMAGE, http=http)
        assert "503" not in str(excinfo.value)


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
            (FakeRegistry(), "registry.example.com/org/agent", 422),
            (FakeRegistry(manifest_body=b"<html>busy</html>"), IMAGE, 422),
            (FakeRegistry(redirect_blobs_to="https://10.0.0.5"), IMAGE, 422),
        ],
    )
    def test_maps_failures_to_status_codes(self, registry: FakeRegistry, image: str, status: int) -> None:
        response = _api_client(registry).post(self.URL, json={"image": image})

        assert response.status_code == status
        assert response.json()["detail"]
