# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signed cookie that authorizes a browser frame to use one connection's proxied UI.

A frame cannot attach a bearer token to the requests its page makes, so Studio
mints this cookie with a normal authenticated call, scoped by ``Path`` to the one
connection's proxy prefix. The proxy checks it on every request.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

COOKIE_NAME = "nhx_observability_ui"


@dataclass(frozen=True)
class UISession:
    principal: str
    workspace: str
    connection: str
    expires_at: int


class SessionSigner:
    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or secrets.token_bytes(32)

    def mint(self, principal: str, workspace: str, connection: str, ttl_seconds: int) -> tuple[str, UISession]:
        session = UISession(principal, workspace, connection, int(time.time()) + ttl_seconds)
        payload = _b64encode(
            json.dumps(
                {"p": session.principal, "w": session.workspace, "c": session.connection, "e": session.expires_at},
                separators=(",", ":"),
            ).encode()
        )
        return f"{payload}.{self._sign(payload)}", session

    def verify(self, token: str | None, connection: str) -> UISession | None:
        """The session in *token* if its signature is valid, it is unexpired, and it names *connection*."""
        if not token or token.count(".") != 1:
            return None
        payload, signature = token.split(".")
        if not hmac.compare_digest(signature, self._sign(payload)):
            return None
        try:
            data = json.loads(_b64decode(payload))
            session = UISession(str(data["p"]), str(data["w"]), str(data["c"]), int(data["e"]))
        except (ValueError, KeyError, TypeError):
            return None
        if session.connection != connection or session.expires_at <= time.time():
            return None
        return session

    def _sign(self, payload: str) -> str:
        return _b64encode(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
