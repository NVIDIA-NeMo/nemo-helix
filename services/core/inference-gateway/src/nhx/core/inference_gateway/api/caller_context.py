# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routers that call models back through the gateway on their caller's behalf, without ever holding its credential.

When the gateway calls a provider marked ``calls_back_as_caller``, it sends the provider its own stored key and a
caller context: the caller's credential sealed with the gateway's key, bound to that provider and valid for a few
minutes. The router returns the context on each call it makes back through the gateway with its key. The gateway
opens it and puts the caller's credential back in place, so access checks and middleware act on the caller. A call
made with a router's key and no valid context is refused.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import os
import time
from collections.abc import Callable, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException, status
from nemo_helix_plugin.models.types import ModelProvider
from nhx.common.auth.bearer import MalformedBearerTokenError
from nhx.common.auth.middleware import API_KEY_HEADER_PATH_PREFIXES, request_bearer_token
from nhx.core.inference_gateway.config import config as inference_gateway_config

logger = logging.getLogger(__name__)

CALLER_CONTEXT_HEADER = "x-nemo-caller-context"
CONTEXT_LIFETIME_SEC = 15 * 60
_VERSION = "v1"
_NONCE_BYTES = 12


class InvalidCallerContextError(Exception):
    """A caller context that is malformed, expired, or was sealed for another provider or with another key."""


class CallerContextSeal:
    """Seals a caller's credential for one router, and opens what that router sends back."""

    def __init__(self, key: bytes, *, clock: Callable[[], float] = time.time) -> None:
        self._aead = AESGCM(key)
        self._clock = clock

    def seal(self, provider_id: str, credential: str | None) -> str:
        payload = json.dumps({"credential": credential, "expires_at": int(self._clock()) + CONTEXT_LIFETIME_SEC})
        nonce = os.urandom(_NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, payload.encode(), _associated_data(provider_id))
        return ".".join((_VERSION, _b64(provider_id.encode()), _b64(nonce + sealed)))

    def open(self, provider_id: str, value: str) -> str | None:
        """The caller credential ``value`` carries for ``provider_id``.

        Raises:
            InvalidCallerContextError: ``value`` wasn't sealed for ``provider_id`` with this key, or has expired.
        """
        if sealed_provider(value) != provider_id:
            raise InvalidCallerContextError("the context was sealed for another provider")
        try:
            sealed = _unb64(value.split(".")[2])
            payload = json.loads(
                self._aead.decrypt(sealed[:_NONCE_BYTES], sealed[_NONCE_BYTES:], _associated_data(provider_id))
            )
        except (InvalidTag, ValueError) as exc:
            raise InvalidCallerContextError("the context wasn't sealed with this gateway's key") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("expires_at"), int):
            raise InvalidCallerContextError("the context is malformed")
        if payload["expires_at"] < self._clock():
            raise InvalidCallerContextError("the context has expired")
        credential = payload.get("credential")
        return credential if isinstance(credential, str) else None


def sealed_provider(value: str) -> str:
    """The ``workspace/name`` of the provider a caller context was sealed for, as it claims.

    Raises:
        InvalidCallerContextError: ``value`` isn't a caller context.
    """
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != _VERSION:
        raise InvalidCallerContextError("the context is malformed")
    try:
        return _unb64(parts[1]).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidCallerContextError("the context is malformed") from exc


def provider_id(provider: ModelProvider) -> str:
    return f"{provider.workspace}/{provider.name}"


def caller_headers(
    headers: Mapping[str, str], router_keys: Mapping[str, str | None], seal: CallerContextSeal
) -> dict[str, str]:
    """``headers`` as the gateway acts on them: a router's call back carries its caller's credential, not its own.

    ``router_keys`` maps each router's ``workspace/name`` to its stored key. A caller context is honoured only
    alongside the key of the router it was sealed for, and dropped otherwise.

    Raises:
        HTTPException: 403 for a call made with a router's key that carries no valid context for it.
    """
    acting = {name.lower(): value for name, value in headers.items()}
    context = acting.pop(CALLER_CONTEXT_HEADER, None)
    credential = _credential(acting)
    if context is not None and (router := _context_router(context, credential, router_keys)) is not None:
        try:
            caller = seal.open(router, context)
        except InvalidCallerContextError as exc:
            raise _refusal(router, str(exc)) from exc
        acting.pop("authorization", None)
        acting.pop("x-api-key", None)
        if caller is not None:
            acting["authorization"] = f"Bearer {caller}"
        return acting
    if credential is not None:
        router = next((router for router, key in router_keys.items() if _same_secret(credential, key)), None)
        if router is not None:
            raise _refusal(router, "it carries none")
    return acting


def refuse_router_keys(headers: Mapping[str, str], router_keys: Mapping[str, str | None]) -> None:
    """Refuse a call that reaches a handler still carrying a router's key: its caller was never put in place.

    Raises:
        HTTPException: 403 for a call made with a router's key.
    """
    credential = _credential({name.lower(): value for name, value in headers.items()})
    if credential is None:
        return
    router = next((router for router, key in router_keys.items() if _same_secret(credential, key)), None)
    if router is not None:
        raise _refusal(router, "the gateway didn't put the caller in its place")


def context_headers(provider: ModelProvider, headers: Mapping[str, str], seal: CallerContextSeal) -> dict[str, str]:
    """The caller context to send ``provider`` with the call ``headers`` describe; none for an ordinary provider."""
    if not provider.calls_back_as_caller:
        return {}
    acting = {name.lower(): value for name, value in headers.items()}
    return {CALLER_CONTEXT_HEADER: seal.seal(provider_id(provider), _credential(acting))}


def _credential(headers: Mapping[str, str]) -> str | None:
    try:
        return request_bearer_token(API_KEY_HEADER_PATH_PREFIXES[0], headers)
    except MalformedBearerTokenError:
        return None


def _context_router(context: str, credential: str | None, router_keys: Mapping[str, str | None]) -> str | None:
    """The router a context names, when the call is made with that router's key."""
    try:
        router = sealed_provider(context)
    except InvalidCallerContextError:
        return None
    if router not in router_keys or not _same_secret(credential, router_keys[router]):
        return None
    return router


def _same_secret(credential: str | None, secret: str | None) -> bool:
    if credential is None or secret is None:
        return credential is None and secret is None
    return hmac.compare_digest(credential.encode(), secret.encode())


def _refusal(router: str, why: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"A call made with the key of model provider {router}, which calls models back as its caller, must carry "
            f"the caller context the gateway sent it, but {why}."
        ),
    )


def _associated_data(provider_id: str) -> bytes:
    return f"{_VERSION}:{provider_id}".encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except binascii.Error as exc:
        raise ValueError("not base64") from exc


_seal: CallerContextSeal | None = None


def gateway_seal() -> CallerContextSeal:
    """The seal built from the gateway's ``caller_context_key``, or a key for this process alone when none is set."""
    global _seal
    if _seal is None:
        configured = inference_gateway_config.caller_context_key
        if configured is None:
            logger.warning(
                "inference_gateway.caller_context_key is unset, so this process uses a key of its own: routers that "
                "call models back as their caller work only while every call reaches this process"
            )
            key = AESGCM.generate_key(bit_length=256)
        else:
            key = base64.b64decode(configured.get_secret_value(), validate=True)
        _seal = CallerContextSeal(key)
    return _seal
