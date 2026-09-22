# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from nhx.common.config import AuthConfig

from .jwt import JWTValidator
from .models import Principal
from .token_claims import TokenClaims

ResolvedTokenKind = Literal["access_key", "oidc_access_token", "workload_access_token", "workload_subject_token"]


def _identity_aliases(subject: str, email: str | None) -> list[str]:
    aliases = [subject]
    if email and email not in aliases:
        aliases.append(email)
    return aliases


def _direct_principal_from_claims(claims: TokenClaims) -> Principal:
    return Principal(
        id=claims.subject,
        email=claims.email,
        groups=claims.groups,
        authz_aliases=_identity_aliases(claims.subject, claims.email),
    )


def _workload_access_principal_from_claims(claims: TokenClaims) -> Principal:
    if claims.actor is None:
        return _direct_principal_from_claims(claims)
    return Principal(
        id=claims.actor.subject,
        groups=claims.actor.groups,
        authz_aliases=[claims.actor.subject],
        on_behalf_of=claims.subject,
        on_behalf_of_email=claims.email,
        on_behalf_of_groups=claims.groups,
        on_behalf_of_authz_aliases=_identity_aliases(claims.subject, claims.email),
    )


@dataclass(frozen=True)
class ResolvedBearerToken:
    claims: TokenClaims
    token_kind: ResolvedTokenKind

    @property
    def principal(self) -> Principal:
        if self.token_kind == "workload_access_token":
            return _workload_access_principal_from_claims(self.claims)
        return _direct_principal_from_claims(self.claims)

    @property
    def scopes(self) -> list[str]:
        return self.claims.scopes

    def principal_headers(self) -> dict[str, str]:
        headers = self.principal.get_headers()
        if self.scopes:
            headers["X-NHX-Scopes"] = " ".join(self.scopes)
        return headers


ExtraBearerTokenResolver = Callable[[str], Awaitable[ResolvedBearerToken | None]]


async def resolve_bearer_token(
    config: AuthConfig,
    token: str,
    *,
    jwt_validator: JWTValidator | None = None,
    extra_resolvers: Sequence[ExtraBearerTokenResolver] = (),
    skip_access_key_check: bool = False,
) -> ResolvedBearerToken | None:
    if config.access_keys.enabled and not skip_access_key_check:
        from .access_keys import validate_access_key_token

        access_key_claims = await validate_access_key_token(config, token)
        if access_key_claims is not None:
            return ResolvedBearerToken(claims=access_key_claims, token_kind="access_key")

    for extra_resolver in extra_resolvers:
        resolved = await extra_resolver(token)
        if resolved is not None:
            return resolved

    if not config.oidc.enabled and not config.allow_unsigned_jwt:
        return None

    validator = jwt_validator or JWTValidator(config)
    oidc_claims = await validator.validate_token(token)
    if oidc_claims is None:
        return None
    return ResolvedBearerToken(claims=oidc_claims, token_kind="oidc_access_token")
