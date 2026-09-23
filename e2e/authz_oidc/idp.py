# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal test OIDC issuer for the authz E2E harness.

Serves the two endpoints ``nhx.common.auth.jwt.JWTValidator`` actually
consumes — ``/.well-known/openid-configuration`` and a JWKS document — over
real HTTP, and mints real RS256-signed JWTs. Token-defect cases (expired,
wrong issuer, wrong audience, unknown signing key, ``alg=none``) are minted
directly; a production IdP cannot produce most of these on demand, which is
why the harness owns its issuer instead of running Dex/Keycloak.

Two RSA keys are generated per session:

- ``KID_ACTIVE`` — published in the JWKS; signs every valid token.
- ``KID_ROGUE`` — never published; signs the "unknown signing key" case.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

KID_ACTIVE = "e2e-active-key"
KID_ROGUE = "e2e-rogue-key"

DEFAULT_AUDIENCE = "nhx-e2e-authz"


@dataclass
class MintSpec:
    """Declarative description of a token to mint (defaults = a valid token)."""

    sub: str
    email: str | None = None
    groups: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    issuer: str | None = None  # None -> the real issuer URL
    audience: str | None = DEFAULT_AUDIENCE
    expires_in: int = 3600  # negative -> already expired
    kid: str = KID_ACTIVE  # KID_ROGUE -> signature by an unpublished key
    unsigned: bool = False  # True -> alg=none token


class MiniOIDCIssuer:
    """A real-HTTP OIDC issuer: discovery + JWKS + RS256 minting."""

    def __init__(self) -> None:
        self._keys = {
            KID_ACTIVE: rsa.generate_private_key(public_exponent=65537, key_size=2048),
            KID_ROGUE: rsa.generate_private_key(public_exponent=65537, key_size=2048),
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.issuer_url: str = ""

    # -- HTTP surface ------------------------------------------------------

    def _jwks_document(self) -> dict:
        # Only the active key is published; the rogue key stays private.
        jwk = json.loads(RSAAlgorithm.to_jwk(self._keys[KID_ACTIVE].public_key()))
        jwk.update({"kid": KID_ACTIVE, "use": "sig", "alg": "RS256"})
        return {"keys": [jwk]}

    def _discovery_document(self) -> dict:
        return {
            "issuer": self.issuer_url,
            "jwks_uri": f"{self.issuer_url}/jwks.json",
            "authorization_endpoint": f"{self.issuer_url}/authorize",
            "token_endpoint": f"{self.issuer_url}/token",
            "device_authorization_endpoint": f"{self.issuer_url}/device",
            "response_types_supported": ["code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "profile", "email"],
        }

    def start(self) -> str:
        issuer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                routes = {
                    "/.well-known/openid-configuration": issuer._discovery_document,
                    "/jwks.json": issuer._jwks_document,
                }
                builder = routes.get(self.path.split("?")[0])
                if builder is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(builder()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server API
                pass  # keep pytest output clean

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.issuer_url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.issuer_url

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self.issuer_url = ""

    # -- Token minting -------------------------------------------------------

    def mint(self, spec: MintSpec) -> str:
        now = int(time.time())
        claims: dict[str, object] = {
            "sub": spec.sub,
            "iat": now - 5,
            "exp": now + spec.expires_in,
            "iss": spec.issuer if spec.issuer is not None else self.issuer_url,
        }
        if spec.audience is not None:
            claims["aud"] = spec.audience
        if spec.email is not None:
            claims["email"] = spec.email
        if spec.groups:
            claims["groups"] = spec.groups
        if spec.scopes:
            claims["scope"] = " ".join(spec.scopes)

        if spec.unsigned:
            # Same wire format the SDK's generate_unsigned_jwt produces:
            # base64url(header).base64url(claims). with an empty signature.
            def b64(obj: dict) -> str:
                raw = json.dumps(obj, separators=(",", ":")).encode()
                return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

            return f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64(claims)}."
        return pyjwt.encode(claims, self._keys[spec.kid], algorithm="RS256", headers={"kid": spec.kid})

    def claims_summary(self, spec: MintSpec) -> str:
        """One-line human description of the minted claims for the audit report."""
        parts = [f"sub={spec.sub}"]
        if spec.email:
            parts.append(f"email={spec.email}")
        if spec.scopes:
            parts.append(f"scope={' '.join(spec.scopes)}")
        if spec.expires_in <= 0:
            parts.append("EXPIRED")
        if spec.issuer is not None:
            parts.append(f"iss={spec.issuer}")
        if spec.audience != DEFAULT_AUDIENCE:
            parts.append(f"aud={spec.audience}")
        if spec.kid != KID_ACTIVE:
            parts.append("key=unpublished")
        if spec.unsigned:
            parts.append("alg=none")
        return ", ".join(parts)
