# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth helpers exposed to plugins.

Plugins must not import ``nhx_common`` directly. This module wraps the pieces of
the platform auth configuration that plugins need. The underlying auth config
lives in ``nhx_common`` (only present in the platform process image), so it is
imported lazily and failures degrade to "disabled" rather than raising in
environments without it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

logger = logging.getLogger(__name__)

_SERVICE_PRINCIPAL_PREFIX = "service:"
_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class RuntimePrincipal(Protocol):
    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def email(self) -> str | None:
        raise NotImplementedError

    @property
    def account_id(self) -> str | None:
        raise NotImplementedError

    @property
    def groups(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def authz_aliases(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def on_behalf_of(self) -> str | None:
        raise NotImplementedError

    @property
    def on_behalf_of_groups(self) -> Sequence[str] | None:
        raise NotImplementedError

    @property
    def on_behalf_of_email(self) -> str | None:
        raise NotImplementedError

    @property
    def on_behalf_of_account_id(self) -> str | None:
        raise NotImplementedError

    @property
    def on_behalf_of_authz_aliases(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def effective_principal(self) -> RuntimePrincipal:
        raise NotImplementedError


class RuntimeAuthPrincipal(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def caller_kind(self) -> str: ...

    @property
    def is_delegated(self) -> bool: ...


class RuntimeAuthClient(Protocol):
    @property
    def auth_enabled(self) -> bool: ...

    @property
    def principal(self) -> RuntimeAuthPrincipal: ...


def current_auth_client(request: Request) -> RuntimeAuthClient:
    from nhx.common.auth.dependencies import get_auth_client

    return get_auth_client(request)


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access."""

    model_config = ConfigDict(from_attributes=True)

    principal_id: str = Field(..., description="The principal's unique identifier")
    principal_account_id: str | None = Field(
        default=None, description="Stable NeMo account identifier for the principal"
    )
    principal_email: str | None = Field(default=None, description="The principal's email address")
    principal_groups: list[str] = Field(default_factory=list, description="Groups the principal belongs to")
    principal_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the principal",
    )
    principal_on_behalf_of: str | None = Field(
        default=None, description="If acting on behalf of another principal, their principal ID"
    )
    principal_on_behalf_of_groups: list[str] | None = Field(
        default=None, description="Groups the on-behalf-of principal belongs to"
    )
    principal_on_behalf_of_email: str | None = Field(
        default=None, description="The on-behalf-of principal's email address"
    )
    principal_on_behalf_of_account_id: str | None = Field(
        default=None,
        description="Stable NeMo account identifier for the on-behalf-of principal",
    )
    principal_on_behalf_of_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the on-behalf-of principal",
    )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AuthContext):
            return super().__eq__(other)
        if isinstance(other, BaseModel):
            return self.model_dump(mode="python") == other.model_dump(mode="python")
        if isinstance(other, Mapping):
            return self.model_dump(mode="python") == dict(other)
        return False

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> AuthContext | None:
        """Create an auth context from validated NeMo principal headers."""
        lower = {key.lower(): value for key, value in headers.items()}
        principal_id = lower.get("x-nhx-principal-id", "").strip()
        if not principal_id:
            return None
        on_behalf_of = lower.get("x-nhx-principal-on-behalf-of", "").strip() or None
        return cls(
            principal_id=principal_id,
            principal_account_id=lower.get("x-nhx-actor-account-id", "").strip() or None,
            principal_email=lower.get("x-nhx-principal-email", "").strip() or None,
            principal_groups=_split_groups(lower.get("x-nhx-principal-groups")),
            principal_authz_aliases=_split_groups(lower.get("x-nhx-actor-aliases")),
            principal_on_behalf_of=on_behalf_of,
            principal_on_behalf_of_groups=_split_groups(lower.get("x-nhx-principal-on-behalf-of-groups"))
            if on_behalf_of
            else None,
            principal_on_behalf_of_email=lower.get("x-nhx-principal-on-behalf-of-email", "").strip() or None
            if on_behalf_of
            else None,
            principal_on_behalf_of_account_id=lower.get("x-nhx-subject-account-id", "").strip() or None
            if on_behalf_of
            else None,
            principal_on_behalf_of_authz_aliases=_split_groups(lower.get("x-nhx-subject-aliases"))
            if on_behalf_of
            else [],
        )

    @classmethod
    def from_principal(cls, principal: RuntimePrincipal) -> Self:
        """Create from a runtime Principal-like value."""
        return cls(
            principal_id=principal.id,
            principal_account_id=principal.account_id,
            principal_email=principal.email,
            principal_groups=list(principal.groups or []),
            principal_authz_aliases=list(principal.authz_aliases or []),
            principal_on_behalf_of=principal.on_behalf_of,
            principal_on_behalf_of_groups=list(principal.on_behalf_of_groups or [])
            if principal.on_behalf_of_groups is not None
            else None,
            principal_on_behalf_of_email=principal.on_behalf_of_email,
            principal_on_behalf_of_account_id=principal.on_behalf_of_account_id,
            principal_on_behalf_of_authz_aliases=list(principal.on_behalf_of_authz_aliases or []),
        )

    @classmethod
    def from_runtime_or_headers(cls, headers: Mapping[str, str]) -> AuthContext | None:
        """Create from the resolved auth runtime.

        Raw principal headers are only trusted after the platform auth middleware
        has validated them and installed the runtime auth context.
        """
        del headers
        return current_auth_context()

    def to_principal(self) -> RuntimePrincipal:
        """Convert to the platform Principal model when nhx-common is available."""
        from nhx.common.auth.models import Principal

        return Principal(
            id=self.principal_id,
            account_id=self.principal_account_id,
            email=self.principal_email,
            groups=self.principal_groups,
            authz_aliases=self.principal_authz_aliases,
            on_behalf_of=self.principal_on_behalf_of,
            on_behalf_of_groups=self.principal_on_behalf_of_groups,
            on_behalf_of_email=self.principal_on_behalf_of_email,
            on_behalf_of_account_id=self.principal_on_behalf_of_account_id,
            on_behalf_of_authz_aliases=self.principal_on_behalf_of_authz_aliases,
        )


def _split_groups(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [group.strip() for group in raw.split(",") if group.strip()]


def current_auth_context() -> AuthContext | None:
    """Return the auth context for the current request/task when running in the platform."""
    try:
        from nhx.common.auth.dependencies import auth_client_context
    except ImportError:
        return None
    auth_client = auth_client_context.get()
    if auth_client is None or auth_client.principal is None or not auth_client.principal.id:
        return None
    return AuthContext.from_principal(auth_client.principal)


def current_workload_delegation_id() -> str | None:
    """ID of the workload delegation record the current request's token was minted from, or None.

    A value is returned only when the platform's middleware authenticated the request this auth context was
    created for with a workload access token the platform signed, that acts for someone, that names the
    delegation record the auth service's token exchange minted it from, and that has not expired since. Nothing
    else fills it: not a header, the body, an access key, a user's own token or a service's token. "The platform
    signed" means the platform's workload signing key, which platform services share: the value is as
    trustworthy as they are.

    None in every other case. To tell "this caller is not a recorded workload" from "nothing here can be
    verified", ask ``workload_delegation_verifiable()``: where that is False (trusted-header mode, or auth off)
    no caller ever has a value. Read this before switching to service credentials.

    It is the record's ID, for ``WorkloadDelegationStore.get_by_id``, not its lookup name. It says which
    delegation the token belongs to. It does not say the record still stands (a token can outlive its record
    until the token expires) or that the peer is the workload itself (a service may forward its caller's token).
    Read the record and compare with ``WorkloadDelegationEntity.is_step_of_job`` or ``is_step_of_attempt``.
    """
    try:
        from nhx.common.auth.dependencies import auth_client_context
    except ImportError:
        return None
    auth_client = auth_client_context.get()
    return None if auth_client is None else auth_client.workload_delegation_id


def workload_delegation_verifiable() -> bool:
    """Whether the current auth context is in the mode in which the platform verifies workload delegations.

    True when that context was made with platform auth on and workload token exchange configured. For a request
    the middleware authenticated, a job step's call then carries a platform-minted delegated token, trusted
    identity headers are refused, and ``current_workload_delegation_id()`` returning None means the request
    carried no unexpired token of a workload the platform recorded a delegation for.

    It reports the mode, not how the caller was authenticated. A context that switched to service credentials
    reports the mode too, also in a background task that no request started, and has no delegation ID. So True
    does not by itself say that a request was authenticated with a token.

    False in trusted-header mode and with auth off, when no auth context exists, and outside the platform
    process: nothing verified says which workload is calling.
    """
    try:
        from nhx.common.auth.dependencies import auth_client_context
    except ImportError:
        return False
    auth_client = auth_client_context.get()
    return auth_client is not None and auth_client.workload_delegation_verifiable


def platform_auth_enabled() -> bool:
    """Return whether platform authentication is enabled.

    Returns ``False`` on any failure to resolve the auth config. The realistic
    failure is ``ImportError``: ``nhx_common`` ships only in the platform process
    image, so when this package is used standalone (outside the platform) there
    is no auth config and "disabled" is the correct answer.

    Other failures are effectively unreachable in the context that matters here
    (the deployment controller, which runs *inside* the platform image): a
    missing config file resolves to defaults (``enabled=False``) rather than
    raising, and a malformed/invalid config file would have already crashed the
    Helix service at startup before any deployment is reconciled. The config
    read is cached from that successful startup load. We therefore accept the
    narrow, largely theoretical fail-open window rather than propagate and block
    deployments on a transient/unexpected error.
    """
    try:
        from nhx.common.config import get_auth_config

        return bool(get_auth_config().enabled)
    except Exception:
        logger.debug("Could not resolve auth config; assuming auth disabled", exc_info=True)
        return False


async def verified_access_key_claims(token: str) -> dict[str, Any] | None:
    """The claims of ``token`` if it is a Scoped Access Key this platform signed and it hasn't expired.

    Applies the platform's own checks: the signature against its access-key signing keys, the algorithm, issuer,
    audience and lifetime. It doesn't check revocation. ``None`` for any other token, when access keys are off,
    and outside the platform process.

    Raises:
        httpx.HTTPError: The platform's signing keys couldn't be fetched.
    """
    try:
        from nhx.common.auth.access_keys import validate_access_key_token
        from nhx.common.config import get_auth_config
    except ImportError:
        return None
    claims = await validate_access_key_token(get_auth_config(), token)
    return None if claims is None else dict(claims.raw_claims)


def inference_gateway_credential(headers: Mapping[str, str]) -> str | None:
    """The credential an Inference Gateway call carries: ``Authorization: Bearer``, else ``x-api-key``.

    Read the way the platform's auth reads it on gateway routes. ``None`` when the call carries neither, when its
    ``Authorization`` header uses Bearer but is malformed, and outside the platform process.
    """
    try:
        from nhx.common.auth.bearer import MalformedBearerTokenError
        from nhx.common.auth.middleware import API_KEY_HEADER_PATH_PREFIXES, request_bearer_token
    except ImportError:
        return None
    lowered = {name.lower(): value for name, value in headers.items()}
    try:
        return request_bearer_token(API_KEY_HEADER_PATH_PREFIXES[0], lowered)
    except MalformedBearerTokenError:
        return None


def is_service_principal_id(principal_id: str) -> bool:
    """Return True for a well-formed Helix service principal.

    When running in the platform process this delegates to the shared parser.
    Standalone plugin installs use the same reserved ``service:<name>`` shape
    validation without requiring ``nhx-common`` to be installed.
    """
    try:
        from nhx.common.auth import is_service_principal
    except ImportError:
        principal = principal_id.strip()
        if not principal.startswith(_SERVICE_PRINCIPAL_PREFIX):
            return False
        return bool(_SERVICE_NAME_RE.fullmatch(principal.removeprefix(_SERVICE_PRINCIPAL_PREFIX)))
    return is_service_principal(principal_id)


__all__ = [
    "AuthContext",
    "RuntimeAuthClient",
    "current_auth_client",
    "current_auth_context",
    "current_workload_delegation_id",
    "inference_gateway_credential",
    "is_service_principal_id",
    "platform_auth_enabled",
    "verified_access_key_claims",
    "workload_delegation_verifiable",
]
