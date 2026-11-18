# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx2 import Response as ClientResponse
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.community_plugins.assets import AssetStore
from nemo_helix_plugin.community_plugins.registry import RegistryStore
from nemo_helix_plugin.dependencies import get_nemo_client
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.platform_runner.community_plugins import installation
from nhx.platform_runner.community_plugins.demo_provider import create_server
from nhx.platform_runner.community_plugins.gateway import ExternalPluginsService
from nhx.studio.service import StudioService

BUNDLE = b"export function Root(){}; export function navItems(){return []}"
SOURCE = {"url": "https://plugins.example/manifest.json"}
BASE = "/apis/plugins/-"


@dataclass
class Rig:
    client: TestClient
    app: FastAPI
    store: RegistryStore
    manifest: dict
    external_calls: list[httpx.Request]
    secret_calls: list[httpx.Request]
    service: ExternalPluginsService
    credentials: dict[str, str]

    def preview(self) -> dict:
        response = self.client.post(f"{BASE}/preview", json=SOURCE)
        assert response.status_code == 200, response.text
        return response.json()

    def install(self) -> ClientResponse:
        reviewed = self.preview()
        return self.client.post(
            f"{BASE}/installed",
            json={
                **SOURCE,
                "manifest_sha256": reviewed["manifest_sha256"],
                "consent": True,
            },
        )


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("nhx.studio.service.discover_plugins", lambda: [])
    manifest = {
        "schema_version": 1,
        "plugin": {
            "name": "tavily",
            "title": "Tavily Demo",
            "endpoints": [
                {
                    "method": "POST",
                    "path": "/v2/workspaces/{workspace}/mcp",
                    "access": "write",
                    "upstream_url": "https://mcp.tavily.com/mcp",
                }
            ],
            "credential": {"secret_name": "tavily-api-key"},
            "bundle": {
                "url": "https://plugins.example/index.js",
                "sha256": hashlib.sha256(BUNDLE).hexdigest(),
                "sidecar": {"contractVersion": "1.0.0", "specifiers": [], "builtAgainst": {}},
            },
            "navigation": [{"label": "Tavily", "path": "/workspaces/{workspace}/plugin/tavily"}],
        },
    }
    external_calls: list[httpx.Request] = []
    secret_calls: list[httpx.Request] = []
    credentials = {"alpha": "key-for-alpha", "beta": "key-for-beta"}

    def external(request: httpx.Request) -> httpx.Response:
        external_calls.append(request)
        if request.url.path == "/manifest.json":
            return httpx.Response(200, json=manifest, headers={"set-cookie": "do-not-forward=secret"})
        if request.url.path == "/index.js":
            return httpx.Response(200, content=BUNDLE, headers={"content-type": "text/javascript"})
        if request.url.path == "/mcp":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {"type": "text", "text": "Local fixture result, not a live Tavily call"},
                        ]
                    },
                },
            )
        return httpx.Response(404)

    def secrets(request: httpx.Request) -> httpx.Response:
        secret_calls.append(request)
        if "/workspaces/denied/" in request.url.path:
            return httpx.Response(403, json={"detail": "access denied"})
        if "/workspaces/missing/" in request.url.path:
            return httpx.Response(404, json={"detail": "not configured"})
        workspace = request.url.path.split("/workspaces/", 1)[1].split("/", 1)[0]
        return httpx.Response(
            200,
            json={
                "name": "tavily-api-key",
                "workspace": workspace,
                "value": credentials.get(workspace, "default-fixture-key"),
            },
        )

    service = ExternalPluginsService(tmp_path / "plugins" / "installed.json")
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(external))
    platform_client = AsyncNemoClient(
        base_url="https://platform.internal",
        default_headers={"authorization": "Bearer caller-platform-token"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(secrets)),
        owns_http_client=True,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await service.on_shutdown()
            await platform_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.dependency_overrides[get_nemo_client] = lambda: platform_client
    app.dependency_overrides[installation.require_plugin_admin] = lambda: "admin@example.com"
    app.include_router(service.get_routers()[0].router, prefix="/apis/plugins")
    StudioService()._configure_plugins(app)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1234)) as client:
        yield Rig(client, app, service.store, manifest, external_calls, secret_calls, service, credentials)


def test_preview_consent_bundle_proxy_disable_and_uninstall(rig: Rig) -> None:
    reviewed = rig.preview()
    assert "not in a sandbox" in reviewed["warning"]
    assert reviewed["manifest"]["plugin"]["credential"]["secret_name"] == "tavily-api-key"
    assert not rig.store.path.exists()
    installed = rig.client.post(
        f"{BASE}/installed",
        json={
            **SOURCE,
            "manifest_sha256": reviewed["manifest_sha256"],
            "consent": True,
        },
    )
    assert installed.status_code == 201, installed.text
    bundle_url = installed.json()["bundleUrl"]
    assert bundle_url.startswith("/plugin-ui/tavily/")
    assert rig.store.path.stat().st_mode & 0o077 == 0
    assert RegistryStore(rig.store.path).load().plugins[0].name == "tavily"  # survives instance recreation
    public = rig.client.get("/apis/plugins").json()
    assert public == [{"name": "tavily", "bundleUrl": bundle_url}]
    assert "credential" not in json.dumps(public)
    bundle = rig.client.get(bundle_url)
    assert bundle.content == BUNDLE
    assert bundle.headers["content-type"].startswith("text/javascript")
    assert rig.client.get(bundle_url, headers={"if-none-match": bundle.headers["etag"]}).status_code == 304
    assert rig.client.get(bundle_url.replace("index.js", "plugin.manifest.json")).json() == {
        "contractVersion": "1.0.0",
        "specifiers": [],
        "builtAgainst": {},
    }
    response = rig.client.post(
        "/apis/plugins/tavily/v2/workspaces/alpha/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        },
        headers={"authorization": "Bearer browser-token", "cookie": "browser=private"},
    )
    assert response.status_code == 200, response.text
    assert rig.secret_calls[-1].url.path == "/apis/secrets/v2/workspaces/alpha/secrets/tavily-api-key/access"
    assert rig.secret_calls[-1].headers["authorization"] == "Bearer caller-platform-token"
    assert rig.external_calls[-1].headers["authorization"] == "Bearer key-for-alpha"
    assert "cookie" not in rig.external_calls[-1].headers
    assert rig.client.patch(f"{BASE}/installed/tavily", json={"enabled": False}).status_code == 204
    assert rig.client.get(bundle_url).status_code == 404
    assert rig.client.get("/apis/plugins").json() == []
    assert rig.client.post("/apis/plugins/tavily/v2/workspaces/alpha/mcp", json={}).status_code == 404
    assert rig.client.patch(f"{BASE}/installed/tavily", json={"enabled": True}).status_code == 204
    assert rig.client.get(bundle_url).status_code == 200
    assert rig.client.delete(f"{BASE}/installed/tavily").status_code == 204
    assert rig.client.get(bundle_url).status_code == 404
    assert rig.store.load().plugins == ()
    assert rig.client.delete(f"{BASE}/installed/tavily").status_code == 404


@pytest.mark.parametrize("consent", [False, "true", 1, None])
def test_explicit_consent_is_required_before_fetch(rig: Rig, consent: object) -> None:
    response = rig.client.post(f"{BASE}/installed", json={**SOURCE, "manifest_sha256": "0" * 64, "consent": consent})
    assert response.status_code == 422
    assert rig.external_calls == []
    assert not rig.store.path.exists()


def test_manifest_change_requires_new_consent(rig: Rig) -> None:
    reviewed = rig.preview()
    rig.manifest["plugin"]["endpoints"][0]["upstream_url"] = "https://different.example/mcp"
    response = rig.client.post(
        f"{BASE}/installed",
        json={
            **SOURCE,
            "manifest_sha256": reviewed["manifest_sha256"],
            "consent": True,
        },
    )
    assert response.status_code == 409
    assert not rig.store.path.exists()
    assert not any(request.url.path == "/index.js" for request in rig.external_calls)


def test_bad_bundle_hash_is_not_installed(rig: Rig) -> None:
    rig.manifest["plugin"]["bundle"]["sha256"] = "0" * 64
    assert rig.install().status_code == 422
    assert not rig.store.path.exists()
    assert not AssetStore(rig.store).root.exists()


def test_duplicate_install_and_core_names_are_rejected(rig: Rig) -> None:
    assert rig.install().status_code == 201
    assert rig.install().status_code == 409
    rig.manifest["plugin"]["name"] = "auth"
    assert rig.install().status_code == 409


def test_cache_tampering_never_serves_changed_bytes(rig: Rig) -> None:
    result = rig.install()
    plugin = rig.store.load().plugins[0]
    assert plugin.bundle is not None
    (AssetStore(rig.store).root / f"{plugin.bundle.sha256}.js").write_bytes(b"tampered")
    response = rig.client.get(result.json()["bundleUrl"])
    assert response.status_code == 503
    assert b"tampered" not in response.content


@pytest.mark.parametrize("workspace,status", [("denied", 403), ("missing", 412)])
def test_secret_access_failure_never_contacts_upstream(rig: Rig, workspace: str, status: int) -> None:
    assert rig.install().status_code == 201
    rig.external_calls.clear()
    response = rig.client.post(f"/apis/plugins/tavily/v2/workspaces/{workspace}/mcp", json={})
    assert response.status_code == status
    assert rig.external_calls == []


def test_workspace_credentials_come_from_path_not_query(rig: Rig) -> None:
    assert rig.install().status_code == 201
    for workspace in ("alpha", "beta"):
        assert (
            rig.client.post(f"/apis/plugins/tavily/v2/workspaces/{workspace}/mcp?workspace=other", json={}).status_code
            == 200
        )
        assert f"/workspaces/{workspace}/" in rig.secret_calls[-1].url.path
        assert rig.external_calls[-1].headers["authorization"] == f"Bearer key-for-{workspace}"


@pytest.mark.parametrize(
    "role,principal,status",
    [
        (False, "user@example.com", 403),
        (True, "service:worker", 403),
        (True, "admin@example.com", 200),
    ],
)
def test_human_helixadmin_guard(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, role: bool, principal: str, status: int
) -> None:
    rig.app.dependency_overrides.pop(installation.require_plugin_admin)
    auth_client = Mock(spec=AuthClient)
    auth_client.auth_enabled = True
    auth_client.principal = Principal(id=principal)
    auth_client.has_role = AsyncMock(return_value=role)
    monkeypatch.setattr(installation, "get_auth_client", lambda request: auth_client)
    assert rig.client.post(f"{BASE}/preview", json=SOURCE).status_code == status
    if status != 200:
        assert rig.external_calls == []


def test_local_admin_is_explicit_loopback_and_same_origin(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.app.dependency_overrides.pop(installation.require_plugin_admin)
    auth_client = Mock(spec=AuthClient)
    auth_client.auth_enabled = False
    monkeypatch.setattr(installation, "get_auth_client", lambda request: auth_client)
    monkeypatch.delenv("NEMO_PLUGINS_LOCAL_ADMIN", raising=False)
    assert rig.client.post(f"{BASE}/preview", json=SOURCE).status_code == 403
    monkeypatch.setenv("NEMO_PLUGINS_LOCAL_ADMIN", "true")
    assert rig.client.post(f"{BASE}/preview", json=SOURCE).status_code == 200
    for headers in ({"origin": "https://evil.example"}, {"host": "evil.example"}, {"x-forwarded-for": "127.0.0.1"}):
        assert rig.client.post(f"{BASE}/preview", json=SOURCE, headers=headers).status_code == 403


def test_demo_lifecycle_against_real_loopback_http_provider(rig: Rig) -> None:
    # Actual bounded HTTP downloads and proxy calls, not MockTransport. The only
    # fixture boundary left is platform secret access (no real provider key).
    rig.credentials["alpha"] = "demo-tavily-key"
    assert rig.client.portal is not None
    assert rig.service.http_client is not None
    rig.client.portal.call(rig.service.http_client.aclose)
    rig.service.http_client = httpx.AsyncClient(trust_env=False, timeout=10)
    with create_server(0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            source = {"url": f"http://127.0.0.1:{server.server_port}/tavily.yaml"}
            reviewed = rig.client.post(f"{BASE}/preview", json=source)
            assert reviewed.status_code == 200, reviewed.text
            installed = rig.client.post(
                f"{BASE}/installed",
                json={
                    **source,
                    "manifest_sha256": reviewed.json()["manifest_sha256"],
                    "consent": True,
                },
            )
            assert installed.status_code == 201, installed.text
            response = rig.client.post(
                "/apis/plugins/tavily/v2/workspaces/alpha/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "tavily_search", "arguments": {"query": "demo"}},
                },
            )
            assert response.status_code == 200, response.text
            assert "LOCAL STUB" in response.json()["result"]["content"][0]["text"]
            # The shipped example is a real built ESM bundle with its real
            # compatibility sidecar; exercise downloading and serving it too.
            source = {"url": f"http://127.0.0.1:{server.server_port}/example.yaml"}
            reviewed = rig.client.post(f"{BASE}/preview", json=source)
            assert reviewed.status_code == 200, reviewed.text
            installed = rig.client.post(
                f"{BASE}/installed",
                json={
                    **source,
                    "manifest_sha256": reviewed.json()["manifest_sha256"],
                    "consent": True,
                },
            )
            assert installed.status_code == 201, installed.text
            bundle = rig.client.get(installed.json()["bundleUrl"])
            expected = reviewed.json()["manifest"]["plugin"]["bundle"]["sha256"]
            assert hashlib.sha256(bundle.content).hexdigest() == expected
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_artifact_fetches_do_not_forward_browser_credentials_or_cookie_jar(rig: Rig) -> None:
    rig.client.headers["authorization"] = "Bearer browser-platform-token"
    rig.client.headers["cookie"] = "browser_session=private"
    assert rig.install().status_code == 201
    for request in rig.external_calls:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers


@pytest.mark.parametrize(
    "status,headers,body,expected",
    [
        (302, {"location": "https://elsewhere.example"}, b"", 502),
        (200, {}, b"too large", 413),
        (200, {"content-encoding": "gzip"}, b"bad compressed bytes", 502),
    ],
)
async def test_invalid_artifact_download_is_rejected(status: int, headers: dict, body: bytes, expected: int) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))
        )
    ) as client:
        with pytest.raises(HTTPException) as error:
            await installation.fetch_artifact(client, SOURCE["url"], 3)
    assert error.value.status_code == expected


async def test_slow_artifact_has_a_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncGenerator[bytes]:
            await asyncio.sleep(1)
            yield b"late"

        async def aclose(self) -> None:
            self.closed = True

    stream = Slow()
    monkeypatch.setattr(installation, "_DOWNLOAD_TIMEOUT_SECONDS", 0.01)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        with pytest.raises(HTTPException) as error:
            await installation.fetch_artifact(client, SOURCE["url"], 100)
    assert error.value.status_code == 504
    assert stream.closed


def test_catalog_is_operator_file_not_remote_discovery(rig: Rig) -> None:
    assert rig.client.get(f"{BASE}/catalog").json() == {"entries": []}
    rig.store.path.parent.mkdir(parents=True, exist_ok=True)
    (rig.store.path.parent / "catalog.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"name": "tavily", "title": "Tavily", **SOURCE},
                ]
            }
        )
    )
    assert rig.client.get(f"{BASE}/catalog").json()["entries"][0]["name"] == "tavily"
    assert rig.external_calls == []
