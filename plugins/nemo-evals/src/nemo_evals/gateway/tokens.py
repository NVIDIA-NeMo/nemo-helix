# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trial tokens: the credentials evaluation trials call the Inference Gateway with, and the model policy they carry.

A trial token names an evaluation trial and the models its agent may call.

With platform auth on, it is a Scoped Access Key evals mints for a user. The auth service signs the claims into it
as the ``evals`` service's attributes, which no user can set. With auth off, it is an unsigned JWT from ``nemo-evals``
carrying the same claim, which anyone can write.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from nemo_evals.gateway.policy import ModelPolicy
from nemo_helix_plugin.auth import platform_auth_enabled, verified_access_key_claims
from nemo_helix_plugin.client.oidc import decode_jwt_claims, decode_jwt_header, generate_unsigned_jwt
from pydantic import ConfigDict, Field, JsonValue, ValidationError

SERVICE_ATTRIBUTES_CLAIM = "nhx_service_attributes"
EVALS_SERVICE = "evals"
UNSIGNED_TOKEN_ISSUER = "nemo-evals"
TOKEN_LIFETIME_SEC = 24 * 60 * 60

REJECTED_TOKEN_CACHE_SEC = 60.0
MAX_CACHED_TOKENS = 4096


class TrialClaims(ModelPolicy):
    """The trial a token was issued to, and the models its agent may call."""

    model_config = ConfigDict(frozen=True)

    v: Literal[1] = 1
    kind: Literal["trial"] = "trial"
    session: str = Field(min_length=1, max_length=256, description="The trial's Intake session.")
    intake_workspace: str = Field(min_length=1, max_length=256)
    evaluation: str | None = Field(default=None, max_length=256)
    task: str | None = Field(default=None, max_length=256)

    def service_attributes(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_token_claims(cls, claims: Mapping[str, Any]) -> TrialClaims | None:
        """The trial a token's claims name, or ``None`` when they name none in a shape this reader knows."""
        attributes = claims.get(SERVICE_ATTRIBUTES_CLAIM)
        evals = attributes.get(EVALS_SERVICE) if isinstance(attributes, Mapping) else None
        if not isinstance(evals, Mapping):
            return None
        try:
            return cls.model_validate(evals)
        except ValidationError:
            return None


def unsigned_token(claims: TrialClaims, *, expires_in_seconds: int = TOKEN_LIFETIME_SEC) -> str:
    """A trial token for a platform with auth off, which has no key to sign one."""
    return generate_unsigned_jwt(
        UNSIGNED_TOKEN_ISSUER,
        issuer=UNSIGNED_TOKEN_ISSUER,
        expires_in_seconds=expires_in_seconds,
        extra_claims={SERVICE_ATTRIBUTES_CLAIM: {EVALS_SERVICE: claims.service_attributes()}},
    )


def claims_a_trial(token: str) -> bool:
    """Whether ``token`` claims to be a trial's, before any check of who issued it."""
    return TrialClaims.from_token_claims(decode_jwt_claims(token)) is not None


TokenVerifier = Callable[[str], Awaitable[Mapping[str, Any] | None]]
"""The claims of a token the platform signed, or ``None`` for any other token."""


def unsigned_token_claims(token: str, *, now: float) -> Mapping[str, Any] | None:
    """The claims of an unexpired, unsigned token from ``nemo-evals``; ``None`` for any other token."""
    if decode_jwt_header(token).get("alg") != "none":
        return None
    claims = decode_jwt_claims(token)
    expires_at = claims.get("exp")
    if claims.get("iss") != UNSIGNED_TOKEN_ISSUER or not isinstance(expires_at, int | float) or expires_at <= now:
        return None
    return claims


class TrialTokenReader:
    """Reads the trial a token names, trusting only tokens ``verify`` accepts.

    Without ``verify``, for a platform with auth off, it trusts unsigned tokens from ``nemo-evals`` instead, and
    no others. Each token's result is remembered until the token expires, or for a minute when it names no trial.
    """

    def __init__(
        self,
        verify: TokenVerifier | None,
        *,
        max_cached_tokens: int = MAX_CACHED_TOKENS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._verify = verify
        self._max_cached_tokens = max_cached_tokens
        self._clock = clock
        self._entries: dict[str, tuple[float, TrialClaims | None]] = {}

    async def trial(self, token: str) -> TrialClaims | None:
        """The trial ``token`` was issued to, or ``None``.

        Raises:
            Exception: Whatever ``verify`` raises when it can't decide, such as the platform's signing keys being
                unreachable. Nothing is remembered then.
        """
        now = self._clock()
        cached = self._entries.get(token)
        if cached is not None and now < cached[0]:
            return cached[1]
        claims = await self._verify(token) if self._verify is not None else unsigned_token_claims(token, now=now)
        trial = TrialClaims.from_token_claims(claims) if claims is not None else None
        expires_at = claims.get("exp") if claims is not None else None
        if trial is None or not isinstance(expires_at, int | float):
            expires_at = now + REJECTED_TOKEN_CACHE_SEC
        self._remember(token, float(expires_at), trial, now)
        return trial

    def _remember(self, token: str, until: float, trial: TrialClaims | None, now: float) -> None:
        self._entries.pop(token, None)
        if len(self._entries) >= self._max_cached_tokens:
            self._entries = {key: entry for key, entry in self._entries.items() if now < entry[0]}
        while len(self._entries) >= self._max_cached_tokens:
            del self._entries[next(iter(self._entries))]
        self._entries[token] = (until, trial)


def platform_token_reader() -> TrialTokenReader:
    """A reader that trusts what this platform issues: signed keys with auth on, unsigned tokens with it off."""
    return TrialTokenReader(verified_access_key_claims if platform_auth_enabled() else None)
