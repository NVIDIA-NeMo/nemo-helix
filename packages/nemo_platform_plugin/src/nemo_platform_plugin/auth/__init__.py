# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth helpers exposed to plugins.

Plugins must not import ``nmp_common`` directly. This module wraps the pieces of
the platform auth configuration that plugins need. The underlying auth config
lives in ``nmp_common`` (only present in the platform process image), so it is
imported lazily and failures degrade to "disabled" rather than raising in
environments without it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

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
        principal_id = lower.get("x-nmp-principal-id", "").strip()
        if not principal_id:
            return None
        on_behalf_of = lower.get("x-nmp-principal-on-behalf-of", "").strip() or None
        return cls(
            principal_id=principal_id,
            principal_account_id=lower.get("x-nmp-actor-account-id", "").strip() or None,
            principal_email=lower.get("x-nmp-principal-email", "").strip() or None,
            principal_groups=_split_groups(lower.get("x-nmp-principal-groups")),
            principal_authz_aliases=_split_groups(lower.get("x-nmp-actor-aliases")),
            principal_on_behalf_of=on_behalf_of,
            principal_on_behalf_of_groups=_split_groups(lower.get("x-nmp-principal-on-behalf-of-groups"))
            if on_behalf_of
            else None,
            principal_on_behalf_of_email=lower.get("x-nmp-principal-on-behalf-of-email", "").strip() or None
            if on_behalf_of
            else None,
            principal_on_behalf_of_account_id=lower.get("x-nmp-subject-account-id", "").strip() or None
            if on_behalf_of
            else None,
            principal_on_behalf_of_authz_aliases=_split_groups(lower.get("x-nmp-subject-aliases"))
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
        """Convert to the platform Principal model when nmp-common is available."""
        from nmp.common.auth.models import Principal

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
        from nmp.common.auth.dependencies import auth_client_context
    except ImportError:
        return None
    auth_client = auth_client_context.get()
    if auth_client is None or auth_client.principal is None or not auth_client.principal.id:
        return None
    return AuthContext.from_principal(auth_client.principal)


def platform_auth_enabled() -> bool:
    """Return whether platform authentication is enabled.

    Returns ``False`` on any failure to resolve the auth config. The realistic
    failure is ``ImportError``: ``nmp_common`` ships only in the platform process
    image, so when this package is used standalone (outside the platform) there
    is no auth config and "disabled" is the correct answer.

    Other failures are effectively unreachable in the context that matters here
    (the deployment controller, which runs *inside* the platform image): a
    missing config file resolves to defaults (``enabled=False``) rather than
    raising, and a malformed/invalid config file would have already crashed the
    platform service at startup before any deployment is reconciled. The config
    read is cached from that successful startup load. We therefore accept the
    narrow, largely theoretical fail-open window rather than propagate and block
    deployments on a transient/unexpected error.
    """
    try:
        from nmp.common.config import get_auth_config

        return bool(get_auth_config().enabled)
    except Exception:
        logger.debug("Could not resolve auth config; assuming auth disabled", exc_info=True)
        return False


def is_service_principal_id(principal_id: str) -> bool:
    """Return True for a well-formed platform service principal.

    When running in the platform process this delegates to the shared parser.
    Standalone plugin installs use the same reserved ``service:<name>`` shape
    validation without requiring ``nmp-common`` to be installed.
    """
    try:
        from nmp.common.auth import is_service_principal
    except ImportError:
        principal = principal_id.strip()
        if not principal.startswith(_SERVICE_PRINCIPAL_PREFIX):
            return False
        return bool(_SERVICE_NAME_RE.fullmatch(principal.removeprefix(_SERVICE_PRINCIPAL_PREFIX)))
    return is_service_principal(principal_id)


__all__ = ["AuthContext", "current_auth_context", "is_service_principal_id", "platform_auth_enabled"]
