# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import ipaddress
import json
import socket
import tarfile
import urllib.request
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import agents as agents_router_module
from nemo_agents_plugin.api.v2.dependencies import get_image_registry_policy, get_registry_http_client
from nemo_agents_plugin.config import DEFAULT_AGENT_IMAGE_REGISTRIES
from nemo_agents_plugin.container import image_bundle
from nemo_agents_plugin.container.image_bundle import (
    ExtractedBundle,
    ImageAccessDeniedError,
    ImageBundleError,
    ImageNotFoundError,
    RegistryPolicy,
    RegistryTimeoutError,
    RegistryUnavailableError,
    read_agent_bundle,
)
from nemo_helix_plugin.container_registry import DOCKER_HUB_REGISTRY

AGENT_YAML = b"config_format: nemo-agents-spec-v1\nname: triage\n"
PUBLIC_ADDRESS = ipaddress.ip_address("93.184.216.34")
CDN_HOST = "cdn.example.net"
IMAGE = "docker.io/sgavrilovic951/sample_agent:0.1"
DEFAULT_POLICY = RegistryPolicy.from_entries(DEFAULT_AGENT_IMAGE_REGISTRIES)


@pytest.fixture(autouse=True)
def resolver(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    answers: dict[str, list[Any]] = {}

    def resolve(host: str) -> list[Any]:
        if host in answers:
            return answers[host]
        try:
            return [ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))]
        except OSError:
            return [PUBLIC_ADDRESS]

    monkeypatch.setattr(image_bundle, "_resolve_host", resolve)
    monkeypatch.setattr(urllib.request, "getproxies", dict)
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
    def __init__(
        self,
        *,
        bundle: dict[str, bytes] | None = None,
        bundle_layer: bytes | None = None,
        env: list[str] | None = None,
        working_dir: str | None = None,
        copy_command: str = "COPY ./ /workspace # buildkit",
        workdir: str | None = None,
        later_copies: list[tuple[str, bytes]] | None = None,
        indexed: bool = True,
        private: bool = False,
        redirect_blobs_to: str | None = None,
        manifest_body: bytes | None = None,
        manifest_status: int | None = None,
        tamper_blobs: bool = False,
    ) -> None:
        bundle = (
            {"workspace/agent.yaml": AGENT_YAML, "workspace/tools/search.py": b"print(1)\n"}
            if bundle is None
            else bundle
        )
        later = later_copies or []
        self.layers = [
            _layer({"etc/os-release": b"ubuntu"}),
            _layer(bundle) if bundle_layer is None else bundle_layer,
            *(layer for _, layer in later),
            _layer({"workspace/.venv/x": b"x"}),
        ]
        container: dict[str, Any] = {
            "Env": env if env is not None else ["PATH=/bin", "AGENT_CONFIG_PATH=/workspace/agent.yaml"]
        }
        if working_dir is not None:
            container["WorkingDir"] = working_dir
        self.config = json.dumps(
            {
                "config": container,
                "history": [
                    {"created_by": "/bin/sh -c #(nop) ADD file:abc in /"},
                    {"created_by": "ENV PORT=8000", "empty_layer": True},
                    *([{"created_by": f"WORKDIR {workdir}", "empty_layer": True}] if workdir else []),
                    {"created_by": copy_command},
                    *({"created_by": command} for command, _ in later),
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
        self.index = json.dumps(
            {
                "manifests": [
                    {"digest": f"sha256:{'0' * 64}", "platform": {"os": "unknown", "architecture": "unknown"}},
                    {"digest": _digest(self.manifest), "platform": {"os": "linux", "architecture": "amd64"}},
                ]
            }
        ).encode()
        self.indexed = indexed
        self.private = private
        self.redirect_blobs_to = redirect_blobs_to
        self.manifest_body = manifest_body
        self.manifest_status = manifest_status
        self.tamper_blobs = tamper_blobs
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
            if self.manifest_status is not None:
                return httpx.Response(self.manifest_status)
            ref = path.rsplit("/", 1)[1]
            top = self.index if self.indexed else self.manifest
            if ref == "0.1" and self.manifest_body is not None:
                return httpx.Response(200, content=self.manifest_body)
            if ref in ("0.1", _digest(top)):
                return httpx.Response(200, content=top)
            if ref == _digest(self.manifest):
                return httpx.Response(200, content=self.manifest)
            return httpx.Response(404)
        digest = path.rsplit("/", 1)[1]
        self.blob_requests.append(digest)
        if self.redirect_blobs_to is not None:
            return httpx.Response(307, headers={"location": f"{self.redirect_blobs_to}/blobs/{digest}"})
        return self._blob(digest)

    def _blob(self, digest: str) -> httpx.Response:
        for blob in (self.config, *self.layers):
            if digest == _digest(blob):
                return httpx.Response(200, content=blob + b"x" if self.tamper_blobs else blob)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


async def _read(image: str, http: httpx.AsyncClient, *registries: str, deadline: float = 120.0) -> ExtractedBundle:
    policy = RegistryPolicy.from_entries(registries) if registries else DEFAULT_POLICY
    return await read_agent_bundle(image, http=http, policy=policy, deadline=deadline)


def _client(
    handle: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


class TestRegistryPolicy:
    @pytest.mark.parametrize("image", ["", "https://docker.io/org/agent", "org agent", "org/agent@sha256:../x"])
    async def test_rejects_what_is_not_a_reference(self, image: str) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="not an image reference"):
                await _read(image, http)

    async def test_refuses_registries_outside_the_allow_list(self) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="not an allowed image registry"):
                await _read("registry.example.com/team/agent", http)

    @pytest.mark.parametrize(
        ("entry", "image"),
        [
            ("docker.io", IMAGE),
            ("Index.Docker.io", "sgavrilovic951/sample_agent:0.1"),
            ("registry-1.docker.io:443", IMAGE),
        ],
    )
    async def test_matches_allow_list_entries_the_way_references_are_normalized(self, entry: str, image: str) -> None:
        async with FakeRegistry().client() as http:
            bundle = await _read(image, http, entry)

        assert "agent.yaml" in bundle.files

    async def test_trusts_a_listed_private_registry(self, resolver: dict[str, list[Any]]) -> None:
        resolver["harbor.corp"] = [ipaddress.ip_address("10.0.0.5")]
        async with FakeRegistry().client() as http:
            bundle = await _read("harbor.corp:5000/team/agent:0.1", http, "harbor.corp:5000")

        assert "agent.yaml" in bundle.files

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
    async def test_wildcard_refuses_registries_on_internal_networks(self, image: str) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="private network"):
                await _read(image, http, "*")

    async def test_wildcard_refuses_names_that_resolve_internally(self, resolver: dict[str, list[Any]]) -> None:
        resolver["registry.example.com"] = [PUBLIC_ADDRESS, ipaddress.ip_address("10.0.0.5")]
        registry = FakeRegistry()
        async with registry.client() as http:
            with pytest.raises(ImageBundleError, match="private network"):
                await _read("registry.example.com/team/agent", http, "*")
        assert registry.requests == []

    async def test_refuses_names_that_do_not_resolve_without_a_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unresolvable(host: str) -> list[Any]:
            raise OSError("no such host")

        monkeypatch.setattr(image_bundle, "_resolve_host", unresolvable)
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="could not be resolved"):
                await _read("registry.example.com/team/agent", http, "*")

    async def test_leaves_resolution_to_the_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unresolvable(host: str) -> list[Any]:
            raise OSError("no such host")

        monkeypatch.setattr(image_bundle, "_resolve_host", unresolvable)
        monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://proxy:3128"})
        async with FakeRegistry().client() as http:
            bundle = await _read(IMAGE, http)

        assert "agent.yaml" in bundle.files

    async def test_honors_no_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unresolvable(host: str) -> list[Any]:
            raise OSError("no such host")

        monkeypatch.setattr(image_bundle, "_resolve_host", unresolvable)
        monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://proxy:3128", "no": ".docker.io"})
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="could not be resolved"):
                await _read(IMAGE, http)

    async def test_reports_a_malformed_host_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def bad_label(host: str) -> list[Any]:
            raise UnicodeError("label too long")

        monkeypatch.setattr(image_bundle, "_resolve_host", bad_label)
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="not a valid host name"):
                await _read("registry.example.com/team/agent", http, "*")

    async def test_refuses_token_services_that_are_not_public_https(self) -> None:
        registry = FakeRegistry()

        def handle(request: httpx.Request) -> httpx.Response:
            response = registry.handle(request)
            if response.status_code == 401:
                response.headers["www-authenticate"] = 'Bearer realm="http://169.254.169.254/token"'
            return response

        async with _client(handle) as http:
            with pytest.raises(ImageBundleError, match="token service"):
                await _read(IMAGE, http)
        assert all(request.url.host != "169.254.169.254" for request in registry.requests)


class TestRedirects:
    async def test_follows_https_blob_redirects_without_forwarding_the_token(self) -> None:
        registry = FakeRegistry(redirect_blobs_to=f"https://{CDN_HOST}")
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert "agent.yaml" in bundle.files
        cdn_requests = [request for request in registry.requests if request.url.host == CDN_HOST]
        assert cdn_requests
        assert all("authorization" not in request.headers for request in cdn_requests)

    @pytest.mark.parametrize("target", [f"http://{CDN_HOST}", "https://10.0.0.5", "https://169.254.169.254"])
    async def test_refuses_redirects_off_public_https(self, target: str) -> None:
        registry = FakeRegistry(redirect_blobs_to=target)
        async with registry.client() as http:
            with pytest.raises(ImageBundleError):
                await _read(IMAGE, http)
        assert all(request.url.host in (DOCKER_HUB_REGISTRY, "auth.docker.io") for request in registry.requests)

    async def test_reports_a_malformed_redirect(self) -> None:
        async with FakeRegistry(redirect_blobs_to="https://[::1").client() as http:
            with pytest.raises(RegistryUnavailableError, match="malformed response"):
                await _read(IMAGE, http)

    async def test_gives_up_after_too_many_redirects(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": str(request.url)})

        async with _client(handle) as http:
            with pytest.raises(RegistryUnavailableError, match="too many times"):
                await _read(IMAGE, http)


class TestIntegrity:
    async def test_pins_the_index_digest_the_tag_resolved_to(self) -> None:
        registry = FakeRegistry()
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.digest == _digest(registry.index)

    async def test_pins_a_single_platform_manifest(self) -> None:
        registry = FakeRegistry(indexed=False)
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.digest == _digest(registry.manifest)

    async def test_reads_by_digest(self) -> None:
        registry = FakeRegistry()
        image = f"docker.io/sgavrilovic951/sample_agent@{_digest(registry.index)}"
        async with registry.client() as http:
            bundle = await _read(image, http)

        assert bundle.digest == _digest(registry.index)

    async def test_refuses_blobs_that_do_not_match_their_digest(self) -> None:
        async with FakeRegistry(tamper_blobs=True).client() as http:
            with pytest.raises(ImageBundleError, match="does not match its digest"):
                await _read(IMAGE, http)

    async def test_refuses_malformed_digests_before_requesting_them(self) -> None:
        manifest = json.dumps({"config": {"digest": "sha256:../../../etc/passwd"}, "layers": []}).encode()
        registry = FakeRegistry(indexed=False, manifest_body=manifest)
        async with registry.client() as http:
            with pytest.raises(ImageBundleError, match="valid sha256 digest"):
                await _read(IMAGE, http)
        assert registry.blob_requests == []

    async def test_reports_malformed_registry_answers(self) -> None:
        async with FakeRegistry(indexed=False, manifest_body=b"<html>busy</html>").client() as http:
            with pytest.raises(ImageBundleError, match="manifest is not JSON"):
                await _read(IMAGE, http)


class TestLimits:
    async def test_caps_the_layers_regardless_of_the_declared_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_LAYER_DOWNLOAD_BYTES", 1024)
        async with FakeRegistry(bundle_layer=b"\0" * 2048).client() as http:
            with pytest.raises(ImageBundleError, match="agent layers exceeds"):
                await _read(IMAGE, http)

    async def test_caps_metadata_bodies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_METADATA_BYTES", 64)
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageBundleError, match="manifest exceeds"):
                await _read(IMAGE, http)

    async def test_caps_how_far_a_layer_inflates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_INFLATED_LAYER_BYTES", 1024 * 1024)
        bomb = _layer({"workspace/agent.yaml": AGENT_YAML, "workspace/.venv/pad.bin": b"\0" * (4 * 1024 * 1024)})
        async with FakeRegistry(bundle_layer=bomb).client() as http:
            with pytest.raises(ImageBundleError, match="inflates to more than"):
                await _read(IMAGE, http)

    async def test_applies_the_staged_bundle_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_bundle, "MAX_ETHOS_STAGED_BYTES", 1000)
        registry = FakeRegistry(bundle={"workspace/agent.yaml": AGENT_YAML, "workspace/big.md": b"x" * 2000})
        async with registry.client() as http:
            with pytest.raises(ImageBundleError, match="larger than 1 KB"):
                await _read(IMAGE, http)

    async def test_rejects_zstd_layers(self) -> None:
        async with FakeRegistry(bundle_layer=b"\x28\xb5\x2f\xfd" + b"\0" * 16).client() as http:
            with pytest.raises(ImageBundleError, match="zstd"):
                await _read(IMAGE, http)

    async def test_gives_up_at_the_deadline(self) -> None:
        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1)
            return httpx.Response(200)

        async with _client(slow) as http:
            with pytest.raises(RegistryTimeoutError, match="took longer than"):
                await _read(IMAGE, http, deadline=0.05)


class TestUpstreamFailures:
    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_reports_registry_failures_as_unavailable(self, status: int) -> None:
        async with FakeRegistry(manifest_status=status).client() as http:
            with pytest.raises(RegistryUnavailableError) as raised:
                await _read(IMAGE, http)
        assert str(status) not in str(raised.value)

    async def test_reports_a_missing_image(self) -> None:
        async with FakeRegistry().client() as http:
            with pytest.raises(ImageNotFoundError):
                await _read("docker.io/sgavrilovic951/sample_agent:9.9", http)

    async def test_reports_a_private_image(self) -> None:
        async with FakeRegistry(private=True).client() as http:
            with pytest.raises(ImageAccessDeniedError, match="private"):
                await _read(IMAGE, http)


class TestBundleContents:
    async def test_returns_the_agent_directory_from_copy_layers_only(self) -> None:
        registry = FakeRegistry()
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.config_path == "/workspace/agent.yaml"
        assert bundle.files == {"agent.yaml": AGENT_YAML, "tools/search.py": b"print(1)\n"}
        assert _digest(registry.layers[0]) not in registry.blob_requests
        assert _digest(registry.layers[-1]) not in registry.blob_requests

    async def test_leaves_out_build_artifacts_and_ignored_files(self) -> None:
        registry = FakeRegistry(
            bundle={
                "workspace/agent.yaml": AGENT_YAML,
                "workspace/Dockerfile.generated": b"FROM x",
                "workspace/.dockerignore": b".env",
                "workspace/nemo_helix-1.0-py3-none-any.whl": b"\x00",
                "workspace/.venv/lib/site.py": b"x",
                "workspace/tools/__pycache__/search.cpython-313.pyc": b"x",
                "workspace/.DS_Store": b"x",
            }
        )
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert list(bundle.files) == ["agent.yaml"]

    async def test_names_a_differently_named_config_agent_yaml(self) -> None:
        registry = FakeRegistry(
            bundle={"app/triage.yaml": AGENT_YAML},
            env=["AGENT_CONFIG_PATH=/app/triage.yaml"],
            copy_command="COPY ./ /app",
        )
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.files == {"agent.yaml": AGENT_YAML}

    async def test_resolves_a_relative_config_path_against_the_working_dir(self) -> None:
        registry = FakeRegistry(env=["AGENT_CONFIG_PATH=agent.yaml"], working_dir="/workspace")
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.config_path == "/workspace/agent.yaml"
        assert "agent.yaml" in bundle.files

    @pytest.mark.parametrize("config_path", ["/agent.yaml", "agent.yaml"])
    async def test_refuses_a_config_in_the_image_root(self, config_path: str) -> None:
        async with FakeRegistry(env=[f"AGENT_CONFIG_PATH={config_path}"]).client() as http:
            with pytest.raises(ImageBundleError, match="image root"):
                await _read(IMAGE, http)

    async def test_requires_agent_config_path(self) -> None:
        async with FakeRegistry(env=["PATH=/bin"]).client() as http:
            with pytest.raises(ImageBundleError, match="AGENT_CONFIG_PATH"):
                await _read(IMAGE, http)

    async def test_takes_only_the_config_directory_from_a_wider_copy(self) -> None:
        registry = FakeRegistry(
            bundle={
                "workspace/configs/agent.yaml": AGENT_YAML,
                "workspace/configs/prompts/system.md": b"be helpful",
                "workspace/pyproject.toml": b"[project]",
            },
            env=["AGENT_CONFIG_PATH=/workspace/configs/agent.yaml"],
        )
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.files == {"agent.yaml": AGENT_YAML, "prompts/system.md": b"be helpful"}

    async def test_resolves_a_relative_copy_against_workdir(self) -> None:
        registry = FakeRegistry(copy_command="COPY . . # buildkit", workdir="/workspace")
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert "agent.yaml" in bundle.files

    async def test_reads_legacy_builder_history(self) -> None:
        registry = FakeRegistry(copy_command="/bin/sh -c #(nop) COPY dir:0123abcd in /workspace")
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert "agent.yaml" in bundle.files

    async def test_overlays_every_copy_into_the_agent_directory(self) -> None:
        registry = FakeRegistry(
            later_copies=[
                ("COPY prompts /workspace/prompts # buildkit", _layer({"workspace/prompts/system.md": b"v2"})),
                ("COPY agent.yaml /workspace/ # buildkit", _layer({"workspace/agent.yaml": b"name: v2\n"})),
            ]
        )
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.files == {
            "agent.yaml": b"name: v2\n",
            "prompts/system.md": b"v2",
            "tools/search.py": b"print(1)\n",
        }

    async def test_honors_whiteouts_in_later_copies(self) -> None:
        registry = FakeRegistry(
            bundle={
                "workspace/agent.yaml": AGENT_YAML,
                "workspace/tools/search.py": b"x",
                "workspace/prompts/old.md": b"x",
            },
            later_copies=[
                (
                    "COPY . /workspace # buildkit",
                    _layer(
                        {
                            "workspace/tools/.wh.search.py": b"",
                            "workspace/prompts/.wh..wh..opq": b"",
                            "workspace/prompts/new.md": b"y",
                        }
                    ),
                )
            ],
        )
        async with registry.client() as http:
            bundle = await _read(IMAGE, http)

        assert bundle.files == {"agent.yaml": AGENT_YAML, "prompts/new.md": b"y"}

    async def test_requires_a_copy_layer(self) -> None:
        async with FakeRegistry(copy_command="RUN true").client() as http:
            with pytest.raises(ImageBundleError, match="No layer copies the agent"):
                await _read(IMAGE, http)

    async def test_requires_agent_yaml_in_the_layers(self) -> None:
        async with FakeRegistry(bundle={"workspace/README.md": b"hi"}).client() as http:
            with pytest.raises(ImageBundleError, match="have no agent.yaml"):
                await _read(IMAGE, http)


def _api_client(registry: FakeRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(agents_router_module.router, prefix="/apis/agents/v2/workspaces/{workspace}")

    async def http() -> Any:
        async with registry.client() as client:
            yield client

    app.dependency_overrides[get_registry_http_client] = http
    app.dependency_overrides[get_image_registry_policy] = lambda: DEFAULT_POLICY
    return TestClient(app, raise_server_exceptions=False)


class TestReadAgentImageBundleRoute:
    URL = "/apis/agents/v2/workspaces/default/agent-images/bundle"

    def test_returns_base64_files_and_the_pinned_image(self) -> None:
        registry = FakeRegistry()
        response = _api_client(registry).post(self.URL, json={"image": f" {IMAGE} "})

        assert response.status_code == 200
        body = response.json()
        assert body["image"] == IMAGE
        assert body["digest"] == _digest(registry.index)
        assert body["pinned_image"] == f"{IMAGE}@{_digest(registry.index)}"
        assert body["config_path"] == "/workspace/agent.yaml"
        assert {item["path"]: base64.b64decode(item["content"]) for item in body["files"]} == {
            "agent.yaml": AGENT_YAML,
            "tools/search.py": b"print(1)\n",
        }

    def test_pins_a_digest_reference_to_itself(self) -> None:
        registry = FakeRegistry()
        image = f"docker.io/sgavrilovic951/sample_agent@{_digest(registry.index)}"

        response = _api_client(registry).post(self.URL, json={"image": image})

        assert response.json()["pinned_image"] == image

    @pytest.mark.parametrize(
        ("registry", "image", "status"),
        [
            (FakeRegistry(), "docker.io/sgavrilovic951/sample_agent:9.9", 404),
            (FakeRegistry(private=True), IMAGE, 422),
            (FakeRegistry(env=[]), IMAGE, 422),
            (FakeRegistry(), "https://hub.docker.com/r/org/agent", 422),
            (FakeRegistry(), "registry.example.com/org/agent", 422),
            (FakeRegistry(indexed=False, manifest_body=b"<html>busy</html>"), IMAGE, 422),
            (FakeRegistry(redirect_blobs_to="https://10.0.0.5"), IMAGE, 422),
        ],
    )
    def test_maps_failures_to_status_codes(self, registry: FakeRegistry, image: str, status: int) -> None:
        response = _api_client(registry).post(self.URL, json={"image": image})

        assert response.status_code == status
        assert response.json()["detail"]

    @pytest.mark.parametrize(("upstream", "status"), [(503, 502), (429, 502)])
    def test_maps_registry_outages_to_gateway_errors(self, upstream: int, status: int) -> None:
        response = _api_client(FakeRegistry(manifest_status=upstream)).post(self.URL, json={"image": IMAGE})

        assert response.status_code == status
