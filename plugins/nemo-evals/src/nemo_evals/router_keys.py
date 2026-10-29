# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router keys: the closed credential a router uses to call models back through the Inference Gateway.

A router registered as a ModelProvider, such as a Switchyard deployment, calls its target models back through the
gateway with the credential its ModelProvider holds. A router key limits that credential to closed-book VirtualModels
the operator names, and the closed-book middleware serves such a router to closed-book evaluations only when it holds
one (see :class:`nemo_evals.gateway.middleware.closed_book.RouterCredentialCheck`). The router itself knows nothing of the policy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nemo_evals.entities import NAME_PATTERN
from nemo_evals.gateway.middleware.closed_book import ends_closed_book
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.tokens import RouterKeyClaims, unsigned_token
from nemo_evals.harbor.trial_credentials import ACCESS_KEY_SCOPE
from nemo_helix_plugin.auth.access_keys.client import AsyncAccessKeysClient
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_helix_plugin.virtual_models.client import AsyncVirtualModelsClient
from pydantic import BaseModel, ConfigDict, Field

ACCESS_KEY_NAME_PREFIX = "nemo-evals-router "
ROUTER_KEY_LIFETIME_SEC = 30 * 24 * 60 * 60


class RouterKeyCreate(BaseModel):
    """A router key to mint and store in a platform secret."""

    model_config = ConfigDict(extra="forbid")

    secret: str = Field(
        pattern=NAME_PATTERN,
        description="Platform secret in the path's workspace to store the key in; created, or its value replaced. "
        "Name it as the router's ModelProvider `api_key_secret_name`.",
    )
    models: list[str] = Field(
        min_length=1,
        description="The closed-book VirtualModels the router may call, as `workspace/name`: its targets and judges.",
    )
    expires_in_seconds: int = Field(default=ROUTER_KEY_LIFETIME_SEC, ge=60, description="The key's lifetime.")


class RouterKey(BaseModel):
    """A router key, stored in a platform secret. The key itself is never returned."""

    secret: str = Field(description="The platform secret holding the key, as `workspace/name`.")
    models: list[str] = Field(description="The VirtualModels the key may call.")
    expires_at: datetime | None = Field(default=None, description="When the key stops working.")
    access_key_id: str | None = Field(
        default=None, description="The Scoped Access Key's id (`jti`), with platform auth on, to revoke it by."
    )


class RouterKeyModelError(ValueError):
    """A model a router key would allow isn't a closed-book VirtualModel the caller can read."""


async def create_router_key(
    *,
    workspace: str,
    request: RouterKeyCreate,
    caller_client: AsyncNemoClient,
    service_client: AsyncHelixClient,
    auth_enabled: bool,
) -> RouterKey:
    """Mint a closed router key for ``request.models`` and store it in ``request.secret``.

    ``caller_client`` reads the models and writes the secret as the caller. ``service_client`` mints the key as the
    evals service on the caller's behalf, since only a platform service can sign its claims.

    Raises:
        RouterKeyModelError: A model isn't a closed-book VirtualModel that names the model it serves.
    """
    models = tuple(dict.fromkeys(request.models))
    for model in models:
        await _require_closed_book(caller_client, model)
    claims = RouterKeyClaims(model_access=ModelAccess.CLOSED, models=models)
    if auth_enabled:
        key = (
            await client_from_platform(service_client, AsyncAccessKeysClient).create_access_key(
                body=AccessKeyCreateRequest(
                    name=f"{ACCESS_KEY_NAME_PREFIX}{workspace}/{request.secret}"[:128],
                    description="Inference Gateway key a router calls closed-book models back through the gateway with.",
                    expires_in_seconds=request.expires_in_seconds,
                    scope=[ACCESS_KEY_SCOPE],
                    service_attributes=claims.service_attributes(),
                )
            )
        ).data()
        token, key_id, expires_at = key.token, key.jti, key.expires_at
    else:
        token = unsigned_token(claims, expires_in_seconds=request.expires_in_seconds)
        key_id, expires_at = None, datetime.now(UTC) + timedelta(seconds=request.expires_in_seconds)
    await _store(caller_client, workspace, request.secret, token)
    return RouterKey(
        secret=f"{workspace}/{request.secret}", models=list(models), expires_at=expires_at, access_key_id=key_id
    )


async def _require_closed_book(client: AsyncNemoClient, model: str) -> None:
    workspace, _, name = model.partition("/")
    if not workspace or not name or "/" in name:
        raise RouterKeyModelError(f"{model!r} isn't a VirtualModel reference of the form workspace/name")
    try:
        virtual_model = (
            await AsyncVirtualModelsClient.from_client(client).get_virtual_model(workspace=workspace, name=name)
        ).data()
    except NotFoundError:
        raise RouterKeyModelError(f"VirtualModel {model} doesn't exist, or you can't read it") from None
    if not ends_closed_book(virtual_model.request_middleware):
        raise RouterKeyModelError(
            f"{model} isn't closed-book: its request_middleware must end with the closed-book middleware"
        )
    if not virtual_model.default_model_entity:
        raise RouterKeyModelError(
            f"{model} sets no default_model_entity, so its caller could name any model through it"
        )


async def _store(client: AsyncNemoClient, workspace: str, name: str, token: str) -> None:
    secrets = AsyncSecretsClient.from_client(client)
    try:
        await secrets.create_secret(
            workspace=workspace,
            body=HelixSecretCreateRequest(
                name=name, value=token, description="nemo-evals router key: closed-book models only."
            ),
        )
    except ConflictError:
        await secrets.update_secret(workspace=workspace, name=name, body=HelixSecretUpdateRequest(value=token))
