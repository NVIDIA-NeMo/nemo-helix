# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth-service client for validating Scoped Access Key lifecycle state."""

import logging
import math
import time

import httpx
import jwt
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.auth.client import AsyncAuthenticationClient
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import (
    AuthenticationError,
    NemoHTTPError,
    NemoResponseValidationError,
    NemoTransportError,
)
from nmp.common.config import AuthConfig
from nmp.common.platform_endpoint import (
    require_authorization_header_endpoint,
    resolve_platform_endpoint,
    resolve_service_endpoint,
)

from .token_claims import TokenClaims
from .token_resolver import ResolvedBearerToken

logger = logging.getLogger(__name__)
ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD = 3
ACCESS_KEY_LIFECYCLE_CIRCUIT_OPEN_SECONDS = 5.0


class AccessKeyLifecycleUnavailableError(RuntimeError):
    """Raised when the auth-service lifecycle validator cannot be trusted."""

    def __init__(self, status_code: int, detail: str, *, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


class AccessKeyLifecycleAuthenticator:
    """Validate access keys through the auth service with fail-closed circuit breaking."""

    def __init__(self, config: AuthConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._http_client = http_client
        self._sdk: AsyncNeMoPlatform | None = None
        self._failure_count = 0
        self._circuit_open_until = 0.0

    def _get_sdk(self) -> AsyncNeMoPlatform:
        if self._sdk is None:
            # Import lazily to avoid an auth -> SDK factory import cycle.
            from nmp.common.sdk_factory import get_async_platform_sdk

            http_client = self._http_client or resolve_platform_endpoint().async_sdk_http_client(follow_redirects=False)
            sdk = get_async_platform_sdk(http_client=http_client)
            self._sdk = sdk.with_options(
                max_retries=0,
                _extra_kwargs={"_strict_response_validation": True},
            )
        return self._sdk

    def _validate_authorization_transport(self) -> None:
        if self._http_client is not None and self._http_client.follow_redirects:
            raise ValueError("Access-key lifecycle HTTP client must not follow redirects")
        require_authorization_header_endpoint(
            resolve_service_endpoint("auth"),
            purpose="Access-key lifecycle validation",
        )

    def _retry_after(self) -> int:
        return max(1, math.ceil(self._circuit_open_until - time.monotonic()))

    def _record_success(self) -> None:
        self._failure_count = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> int | None:
        self._failure_count += 1
        if self._failure_count >= ACCESS_KEY_LIFECYCLE_CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open_until = time.monotonic() + ACCESS_KEY_LIFECYCLE_CIRCUIT_OPEN_SECONDS
            return self._retry_after()
        return None

    def _unavailable(self, status_code: int, detail: str) -> AccessKeyLifecycleUnavailableError:
        return AccessKeyLifecycleUnavailableError(
            status_code,
            detail,
            retry_after=self._record_failure(),
        )

    async def authenticate(self, token: str) -> ResolvedBearerToken | None:
        """Return trusted access-key claims, or None when auth rejects the token."""
        now = time.monotonic()
        if now < self._circuit_open_until:
            retry_after = self._retry_after()
            logger.warning("Access-key lifecycle validation circuit is open for %s more seconds", retry_after)
            raise AccessKeyLifecycleUnavailableError(
                503,
                "Access-key lifecycle validation unavailable",
                retry_after=retry_after,
            )
        if self._circuit_open_until > 0.0:
            # Timer just lapsed — reset so the N-failure threshold applies fresh to this probe.
            self._failure_count = 0
            self._circuit_open_until = 0.0

        try:
            self._validate_authorization_transport()
            auth_client = (
                client_from_platform(self._get_sdk(), AsyncAuthenticationClient)
                .with_headers({"Authorization": f"Bearer {token}"})
                .with_options(timeout=self._config.policy_decision_point_request_timeout_seconds)
            )
            result = (await auth_client.authenticate_bearer_token_get()).data()
        except ValueError as exc:
            logger.error("Access-key lifecycle validation is misconfigured: %s", exc)
            raise self._unavailable(503, "Access-key lifecycle validation unavailable") from exc
        except AuthenticationError:
            self._record_success()
            return None
        except NemoTransportError as exc:
            request_url = exc.request.url if exc.request is not None else "<unknown>"
            if isinstance(exc.error, httpx.TimeoutException):
                logger.error("Access-key lifecycle validation timed out at %s: %s", request_url, exc)
                raise self._unavailable(504, "Access-key lifecycle validation timeout") from exc
            logger.error("Cannot connect to access-key lifecycle validator at %s: %s", request_url, exc)
            raise self._unavailable(503, "Access-key lifecycle validation unavailable") from exc
        except (NemoHTTPError, NemoResponseValidationError) as exc:
            logger.error("Access-key lifecycle validation failed at %s: %s", exc.http_response.request.url, exc)
            raise self._unavailable(503, "Access-key lifecycle validation unavailable") from exc

        if result.token_kind != "access_key":
            self._record_success()
            return None
        if not result.jti:
            logger.error("Access-key lifecycle validator returned an invalid success response")
            raise self._unavailable(503, "Access-key lifecycle validation unavailable")

        self._record_success()
        try:
            unverified_payload = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                    "verify_aud": False,
                },
            )
        except jwt.PyJWTError:
            unverified_payload = {}
        return ResolvedBearerToken(
            claims=TokenClaims(
                subject=result.principal,
                email=result.email,
                groups=result.groups or [],
                scopes=result.scopes or [],
                raw_claims={
                    **unverified_payload,
                    "nmp_token_type": "access_key",
                    "jti": result.jti,
                    "sub": result.principal,
                    **({"email": result.email} if result.email is not None else {}),
                },
            ),
            token_kind="access_key",
        )
