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
# Workload access tokens carry the Scoped Access Key ID of their subject, so a delegated call keeps the key's limits.
WORKLOAD_ACCESS_KEY_JTI_CLAIM = "nhx_access_key_jti"


def _identity_aliases(subject: str, email: str | None) -> list[str]:
    aliases = [subject]
    if email and email not in aliases:
        aliases.append(email)
    return aliases


def _string_claim(claims: TokenClaims, name: str) -> str | None:
    value = claims.raw_claims.get(name)
    return value if isinstance(value, str) and value else None


def _direct_principal_from_claims(claims: TokenClaims, *, access_key_jti: str | None = None) -> Principal:
    return Principal(
        id=claims.subject,
        email=claims.email,
        groups=claims.groups,
        authz_aliases=_identity_aliases(claims.subject, claims.email),
        access_key_jti=access_key_jti,
    )


def _workload_access_principal_from_claims(claims: TokenClaims) -> Principal:
    if claims.actor is None:
        account_id = claims.raw_claims.get("nhx_actor_account_id")
        aliases = claims.raw_claims.get("nhx_actor_aliases")
        if (
            isinstance(account_id, str)
            and account_id
            and isinstance(aliases, list)
            and all(isinstance(alias, str) for alias in aliases)
        ):
            return Principal(
                id=claims.subject,
                account_id=account_id,
                email=claims.email,
                groups=claims.groups,
                authz_aliases=[alias for alias in aliases if isinstance(alias, str)],
                access_key_jti=_string_claim(claims, WORKLOAD_ACCESS_KEY_JTI_CLAIM),
            )
        return _direct_principal_from_claims(
            claims, access_key_jti=_string_claim(claims, WORKLOAD_ACCESS_KEY_JTI_CLAIM)
        )
    return Principal(
        id=claims.actor.subject,
        groups=claims.actor.groups,
        authz_aliases=[claims.actor.subject],
        on_behalf_of=claims.subject,
        on_behalf_of_email=claims.email,
        on_behalf_of_groups=claims.groups,
        on_behalf_of_authz_aliases=_identity_aliases(claims.subject, claims.email),
        on_behalf_of_access_key_jti=_string_claim(claims, WORKLOAD_ACCESS_KEY_JTI_CLAIM),
    )


@dataclass(frozen=True)
class ResolvedBearerToken:
    claims: TokenClaims
    token_kind: ResolvedTokenKind

    @property
    def principal(self) -> Principal:
        if self.token_kind == "workload_access_token":
            return _workload_access_principal_from_claims(self.claims)
        if self.token_kind == "access_key":
            return _direct_principal_from_claims(self.claims, access_key_jti=_string_claim(self.claims, "jti"))
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
        from .access_keys import (
            access_key_has_workspace_limit,
            is_access_key_token_candidate,
            validate_access_key_token,
        )

        access_key_claims = await validate_access_key_token(config, token)
        if access_key_claims is not None:
            if config.policy_decision_point_provider == "opa" and access_key_has_workspace_limit(access_key_claims):
                # An external OPA never sees the auth service's workspace-limit lookup, so it can't enforce the
                # limit; refuse the key rather than let it act unlimited.
                return None
            return ResolvedBearerToken(claims=access_key_claims, token_kind="access_key")
        # validate_access_key_token returns None both for tokens that are not access keys and for
        # access keys that fail validation (bad signature, expired, wrong issuer). Only the former
        # should fall through to the workload and OIDC resolvers; a failed access key is rejected
        # here rather than spending their IdP round-trips on a token that can only be refused.
        if is_access_key_token_candidate(token):
            return None

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
