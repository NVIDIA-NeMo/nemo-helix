# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.tokens import RouterKeyClaims, evals_claims
from nemo_evals.router_keys import RouterKeyCreate, RouterKeyModelError, create_router_key
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.oidc import decode_jwt_claims

VMS = "/apis/inference-gateway/v2/workspaces/models/virtual-models/"
SECRETS = "/apis/secrets/v2/workspaces/routers/secrets"
ACCESS_KEYS = "/apis/auth/v2/access-keys"
CLOSED_BOOK = {"name": "nemo-evals-closed-book", "config_type": "closed_book", "config": {}}
NOW = "2026-10-29T12:00:00Z"


def _vm(name: str, *, closed: bool = True, default: str | None = "models/served") -> dict[str, Any]:
    return {
        "workspace": "models",
        "name": name,
        "default_model_entity": default,
        "request_middleware": [CLOSED_BOOK] if closed else [],
        "created_at": NOW,
        "updated_at": NOW,
    }


class Platform:
    """Serves VirtualModels, secrets and access keys, keeping every request it saw."""

    def __init__(self, *, secret_exists: bool = False) -> None:
        self.vms = {
            "cb-strong": _vm("cb-strong"),
            "cb-weak": _vm("cb-weak"),
            "open-strong": _vm("open-strong", closed=False),
            "unpinned": _vm("unpinned", default=None),
        }
        self.secret_exists = secret_exists
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        path = request.url.path
        if request.method == "GET" and path.startswith(VMS):
            vm = self.vms.get(path.removeprefix(VMS))
            return httpx.Response(200, json=vm) if vm else httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and path == SECRETS:
            if self.secret_exists:
                return httpx.Response(409, json={"detail": "exists"})
            return httpx.Response(201, json=self._secret())
        if request.method == "PATCH" and path == f"{SECRETS}/sy-hop":
            return httpx.Response(200, json=self._secret())
        if request.method == "POST" and path == ACCESS_KEYS:
            return httpx.Response(
                200,
                json={
                    "jti": "ak_1",
                    "principal": "user@example.com",
                    "status": "ACTIVE",
                    "issuer": "nemo",
                    "audiences": ["nemo"],
                    "scope": ["inference"],
                    "created_at": NOW,
                    "expires_at": "2026-11-28T12:00:00Z",
                    "token": "minted-router-key",
                    "token_type": "Bearer",
                },
            )
        return httpx.Response(500, json={"detail": f"unexpected {request.method} {path}"})

    def _secret(self) -> dict[str, Any]:
        return {"workspace": "routers", "name": "sy-hop", "created_at": NOW, "updated_at": NOW}

    def bodies(self, method: str, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.seen if r.method == method and r.url.path == path]


@pytest.fixture
async def client_for() -> AsyncIterator[Any]:
    clients: list[httpx.AsyncClient] = []

    def make(platform: Platform) -> AsyncNemoClient:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(platform))
        clients.append(http_client)
        return AsyncNemoClient(base_url="http://platform.test", http_client=http_client)

    yield make
    for http_client in clients:
        await http_client.aclose()


async def test_with_auth_off_the_key_is_an_unsigned_closed_router_key_stored_in_the_secret(client_for: Any) -> None:
    platform = Platform()
    client = client_for(platform)

    key = await create_router_key(
        workspace="routers",
        request=RouterKeyCreate(secret="sy-hop", models=["models/cb-strong", "models/cb-weak", "models/cb-strong"]),
        caller_client=client,
        service_client=client,
        auth_enabled=False,
    )

    [stored] = platform.bodies("POST", SECRETS)
    assert stored["name"] == "sy-hop"
    assert evals_claims(decode_jwt_claims(stored["value"])) == RouterKeyClaims(
        model_access=ModelAccess.CLOSED, models=("models/cb-strong", "models/cb-weak")
    )
    assert (key.secret, key.models, key.access_key_id) == (
        "routers/sy-hop",
        ["models/cb-strong", "models/cb-weak"],
        None,
    )
    assert "value" not in key.model_dump()


async def test_with_auth_on_the_evals_service_mints_a_signed_key_and_an_existing_secret_is_replaced(
    client_for: Any,
) -> None:
    platform = Platform(secret_exists=True)
    client = client_for(platform)

    key = await create_router_key(
        workspace="routers",
        request=RouterKeyCreate(secret="sy-hop", models=["models/cb-strong"], expires_in_seconds=3600),
        caller_client=client,
        service_client=client,
        auth_enabled=True,
    )

    [minted] = platform.bodies("POST", ACCESS_KEYS)
    assert minted["service_attributes"] == {
        "model_access": "closed",
        "models": ["models/cb-strong"],
        "v": 1,
        "kind": "router",
    }
    assert (minted["scope"], minted["expires_in_seconds"]) == (["inference"], 3600)
    assert platform.bodies("PATCH", f"{SECRETS}/sy-hop") == [{"value": "minted-router-key"}]
    assert key.access_key_id == "ak_1"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        pytest.param("models/open-strong", "isn't closed-book", id="an open model"),
        pytest.param("models/unpinned", "sets no default_model_entity", id="a model that lets its caller pick"),
        pytest.param("models/missing", "doesn't exist", id="a missing model"),
        pytest.param("cb-strong", "workspace/name", id="a bare name"),
    ],
)
async def test_a_key_is_only_minted_for_closed_book_models(client_for: Any, model: str, message: str) -> None:
    platform = Platform()
    client = client_for(platform)

    with pytest.raises(RouterKeyModelError, match=message):
        await create_router_key(
            workspace="routers",
            request=RouterKeyCreate(secret="sy-hop", models=["models/cb-strong", model]),
            caller_client=client,
            service_client=client,
            auth_enabled=True,
        )

    assert platform.bodies("POST", ACCESS_KEYS) == []
    assert platform.bodies("POST", SECRETS) == []
