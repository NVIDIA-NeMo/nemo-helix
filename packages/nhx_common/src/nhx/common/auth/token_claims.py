# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth claim projection from decoded JWT payloads."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from nhx.common.config import AuthConfig

from .json_payload import JsonObject

logger = logging.getLogger(__name__)

#: Where the auth service's token exchange names the workload delegation a delegated token was minted from.
WORKLOAD_DELEGATION_ID_CLAIM_PATH: tuple[str, str, str] = ("nhx_service_attributes", "auth", "delegation_id")

#: Clock skew allowed when a workload access token's time claims are checked.
WORKLOAD_ACCESS_TOKEN_LEEWAY_SECONDS = 30


@dataclass
class ActorClaims:
    """Validated RFC 8693 actor claims."""

    subject: str
    groups: list[str]


@dataclass
class TokenClaims:
    """Validated token claims."""

    subject: str
    email: str | None
    groups: list[str]
    scopes: list[str]
    raw_claims: JsonObject
    actor: ActorClaims | None = None


def groups_from_claim(value: object) -> list[str]:
    """Parse a groups JWT claim that may be a comma-separated string, list, or role map."""
    if isinstance(value, str):
        return [g.strip() for g in value.split(",") if g.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, dict):
        return [key.strip() for key in value if isinstance(key, str) and key.strip()]
    return []


def scopes_from_claim(value: object) -> list[str]:
    """Parse a scopes JWT claim that may be whitespace-delimited or a list."""
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def workload_delegation_id_from_claims(claims: Mapping[str, object]) -> str | None:
    """The delegation ID in ``claims``: None unless each level is a mapping and the leaf a non-empty string.

    Says nothing about who wrote the claims. Callers must have verified the token and its kind first.
    """
    found: object = claims
    for key in WORKLOAD_DELEGATION_ID_CLAIM_PATH:
        found = found.get(key) if isinstance(found, Mapping) else None
    return found if isinstance(found, str) and found else None


class TokenClaimsExtractor:
    """Project decoded JWT claim objects into explicit auth claim types."""

    def __init__(self, config: AuthConfig):
        self.config = config

    def extract(self, claims: JsonObject) -> TokenClaims | None:
        """Extract principal information and scopes from claims."""
        subject = claims.get(self.config.oidc.subject_claim, claims.get("sub"))
        if not isinstance(subject, str) or not subject:
            logger.warning("Token is missing a valid subject claim")
            return None

        email_value = claims.get(self.config.oidc.email_claim)
        email = email_value if isinstance(email_value, str) else None

        scopes = scopes_from_claim(claims.get("scope") or claims.get("scp"))
        prefix = self.config.oidc.effective_scope_prefix
        if prefix:
            scopes = [scope.removeprefix(prefix) for scope in scopes]

        return TokenClaims(
            subject=subject,
            email=email,
            groups=self.groups_from_claims(claims),
            scopes=scopes,
            raw_claims=claims,
            actor=self.actor_from_claims(claims),
        )

    def groups_from_claims(self, claims: JsonObject) -> list[str]:
        """Extract normalized groups from configured or legacy fallback claims."""
        for claim_name in [self.config.oidc.groups_claim, "cognito:groups"]:
            if claim_name not in claims:
                continue
            return groups_from_claim(claims[claim_name])
        return []

    def actor_from_claims(self, claims: JsonObject) -> ActorClaims | None:
        """Extract RFC 8693 act claims when a valid actor subject is present."""
        actor_claims_value = claims.get("act")
        if not isinstance(actor_claims_value, dict):
            return None
        actor_claims = cast(JsonObject, actor_claims_value)

        actor_subject = actor_claims.get("sub")
        if not isinstance(actor_subject, str):
            return None

        actor_subject = actor_subject.strip()
        if not actor_subject:
            return None

        return ActorClaims(
            subject=actor_subject,
            groups=self.groups_from_claims(actor_claims),
        )
