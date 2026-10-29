# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed requests to associate OpenShell provider credentials with Helix secrets."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import uuid
from collections.abc import Callable

from nemo_helix_plugin.openshell.types import GatewayWorkload
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

MAX_TICKET_LIFETIME = 300
MAX_BINDING_LIFETIME = 3600
TICKET_PREFIX = "nemo-registration-v1"


class RegistrationDenied(ValueError):
    """Reject registration or credential lookup without exposing request contents."""


class BindingScope(BaseModel):
    """Describe which provider slot may use a Helix secret, for which creator and workload.

    Jobs and Deployments OpenShell backends construct this data before signing a
    registration request. Signing adds the request ID and timestamps in TicketClaims.

    The driver uses these fields to match the registering provider and, on later
    resolution, identify the secret and check creator access and workload state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_min_length=1, str_max_length=256)

    gateway_id: str
    gateway_workspace: str
    provider: str
    credential_key: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    secret_workspace: str = Field(pattern=r"^[^/\s]+$")
    secret_name: str = Field(pattern=r"^[^/\s]+$")
    job_id: str
    # None is an explicit auth-disabled binding, accepted only by a driver
    # configured for that mode. It is covered by the ticket signature.
    principal_id: str | None = None
    workload: GatewayWorkload | None = None


class TicketClaims(BindingScope):
    """Registration scope with a unique request ID and validity timestamps.

    ``expires_at`` limits registration. ``binding_expires_at`` limits later
    lookups for standalone bindings; workload bindings use live lifecycle checks.
    """

    ticket_id: str
    issued_at: int
    expires_at: int
    binding_expires_at: int


class RegistrationTickets:
    """Sign and verify requests to bind OpenShell providers to existing Helix secrets.

    Jobs and Deployments OpenShell backends send these requests through OpenShell's
    credential-value field. The credential driver verifies each request before
    storing the provider-to-secret association and returning an opaque handle.

    A request contains the secret reference, provider slot, creator, workload
    and expiry times. This integration calls that signed request a ticket.
    """

    def __init__(self, key: bytes, gateway_id: str, *, clock: Callable[[], float] = time.time) -> None:
        if len(key) < 32 or not gateway_id:
            raise ValueError("A dedicated signing key of at least 32 bytes and a gateway ID are required.")
        self._key = key
        self.gateway_id = gateway_id
        self.clock = clock

    def sign(
        self,
        scope: BindingScope,
        *,
        binding_lifetime: int = MAX_BINDING_LIFETIME,
    ) -> SecretStr:
        """Sign a registration request after the Jobs or Deployments backend approves it.

        This operation performs no platform calls. The backend must check the
        creator, secret, workload and injection profile before signing.
        """
        if scope.gateway_id != self.gateway_id or not 1 <= binding_lifetime <= MAX_BINDING_LIFETIME:
            raise RegistrationDenied("Invalid registration scope or lifetime.")
        now = int(self.clock())
        claims = TicketClaims(
            **scope.model_dump(),
            ticket_id=uuid.uuid4().hex,
            issued_at=now,
            expires_at=now + min(MAX_TICKET_LIFETIME, binding_lifetime),
            binding_expires_at=now + binding_lifetime,
        )
        payload = base64.urlsafe_b64encode(claims.model_dump_json().encode()).decode().rstrip("=")
        message = f"{TICKET_PREFIX}.{payload}"
        signature = hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()
        return SecretStr(f"{message}.{signature}")

    def verify(self, ticket: SecretStr) -> TicketClaims:
        """Return the signed scope after checking the signature, gateway and time limits."""
        try:
            raw = ticket.get_secret_value()
            if len(raw) > 8192:
                raise ValueError
            prefix, payload, signature = raw.split(".")
            expected = hmac.new(self._key, f"{prefix}.{payload}".encode(), hashlib.sha256).hexdigest()
            if prefix != TICKET_PREFIX or not hmac.compare_digest(expected, signature):
                raise ValueError
            claims = TicketClaims.model_validate_json(
                base64.b64decode(payload + "=" * (-len(payload) % 4), altchars=b"-_", validate=True)
            )
            now = int(self.clock())
            if (
                claims.gateway_id != self.gateway_id
                or not claims.issued_at <= now < claims.expires_at <= claims.binding_expires_at
                or not 0 < claims.expires_at - claims.issued_at <= MAX_TICKET_LIFETIME
                or not 0 < claims.binding_expires_at - claims.issued_at <= MAX_BINDING_LIFETIME
            ):
                raise ValueError
            return claims
        except (ValueError, ValidationError, UnicodeError, TypeError):
            raise RegistrationDenied("Invalid or expired registration ticket.") from None
