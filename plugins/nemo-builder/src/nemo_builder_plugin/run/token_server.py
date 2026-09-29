# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build token-server`` -- a LOCAL-ONLY stand-in for a registry's token service.

**Not part of the design.** RFC 001 Requirement 5 keeps this system from operating a registry, and
a real deployment's registry brings its own token service: GAR's, Harbor's, Artifactory's. The
minikube quickstart runs ``distribution``, which delegates token issuance to a realm it is
configured with, and this is that realm -- so that the credential broker's token-specification
client runs against a token service locally, rather than only in production.

It plays a token service's part and nothing more, as the Distribution token specification
describes one:

- **It authenticates the caller.** Callers present a Kubernetes ServiceAccount token as the Basic
  password, bound to an audience only this service accepts -- the way workload identity lets a
  pod authenticate to a cloud registry. ``TokenReview`` says who it is.
- **It grants the intersection** of the scope asked for and what the caller's entry in ``grants``
  allows. Asking for more is not an error: the rest is left out, and the registry refuses what
  the token does not cover.
- **It issues a short-lived ES256 JWT** whose ``access`` claim is the whole grant. The registry
  trusts it through a JWKS derived from the same key (``nhx-build token-server jwks``), so the two
  cannot disagree about the key id.

In the quickstart it grants the broker pull and push, and a reader pull, to check results with.
A push step presenting its own token here gets nothing: it has to go through the broker.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from nemo_helix_plugin.log_utils import sanitize_for_log
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: The only actions ever granted. `delete` and `*` are never granted: nothing this system does
#: needs them.
_GRANTABLE = ("pull", "push")


class Grant(BaseModel):
    """What one subject may be granted."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="The ServiceAccount's username, `system:serviceaccount:<ns>:<name>`.")
    repositories: str = Field(
        default="", description="A repository prefix it is limited to; empty for every repository."
    )
    actions: list[str] = Field(default_factory=lambda: ["pull"])


class TokenServerConfig(BaseModel):
    """Read from the YAML file ``NHX_TOKEN_SERVER_CONFIG`` names."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(description="The registry's `auth.token.service`; tokens are issued for it alone.")
    issuer: str = Field(default="nhx-local-token-server", description="The registry's `auth.token.issuer`.")
    key_file: str = Field(description="PKCS#8 PEM, P-256: the key tokens are signed with.")
    audience: str = Field(
        default="nhx-local-token-server",
        description="The audience a caller's ServiceAccount token must be bound to, so it is useless anywhere else.",
    )
    ttl_seconds: int = Field(default=900, ge=60, le=3600)
    port: int = 8080
    grants: list[Grant] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Scope:
    """One ``repository:<name>:<actions>`` item of a token request, or of a grant."""

    name: str
    actions: tuple[str, ...]

    def as_access(self) -> dict[str, object]:
        """The ``access`` claim entry the token specification defines."""
        return {"type": "repository", "name": self.name, "actions": list(self.actions)}


def parse_scope(value: str) -> Scope | None:
    """``repository:ws/app:pull,push`` -> Scope, or None for anything that is not a repository scope.

    Split from the right, because the name is the only part that may contain a colon. A scope
    that is not a plain ``repository`` -- ``registry:catalog:*`` and the like -- is ignored rather
    than refused: the right answer is "not granted", which leaving it out says.
    """
    type_and_name, sep, actions = value.rpartition(":")
    if not sep:
        return None
    kind, sep, name = type_and_name.partition(":")
    if not sep or kind != "repository" or not name:
        return None
    wanted = tuple(dict.fromkeys(action.strip() for action in actions.split(",") if action.strip()))
    return Scope(name=name, actions=wanted) if wanted else None


def grant(subject: str, requested: list[Scope], grants: list[Grant]) -> list[Scope]:
    """The part of ``requested`` that ``subject``'s entry allows. Never more than was asked for."""
    entry = next((g for g in grants if g.subject == subject), None)
    if entry is None:
        return []
    prefix = entry.repositories.strip("/")
    granted: list[Scope] = []
    for scope in requested:
        if prefix and not scope.name.startswith(prefix + "/"):
            continue
        actions = tuple(a for a in scope.actions if a in entry.actions and a in _GRANTABLE)
        if actions:
            granted.append(Scope(name=scope.name, actions=actions))
    return granted


# --- Tokens ----------------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def public_jwk(public_key: ec.EllipticCurvePublicKey) -> dict[str, str]:
    """The public key as a JWK, without ``kid`` -- the members RFC 7638 hashes, and nothing else."""
    if not isinstance(public_key.curve, ec.SECP256R1):
        raise ValueError("tokens are signed with ES256, which is P-256 only")
    numbers = public_key.public_numbers()
    return {
        "crv": "P-256",
        "kty": "EC",
        "x": _b64url(numbers.x.to_bytes(32, "big")),
        "y": _b64url(numbers.y.to_bytes(32, "big")),
    }


def key_id(public_key: ec.EllipticCurvePublicKey) -> str:
    """The RFC 7638 thumbprint: derived from the key, so the JWKS and the tokens agree without a
    second value to keep in sync."""
    canonical = json.dumps(public_jwk(public_key), separators=(",", ":"), sort_keys=True)
    return _b64url(hashlib.sha256(canonical.encode()).digest())


def jwks(public_key: ec.EllipticCurvePublicKey) -> dict[str, list[dict[str, str]]]:
    """What the registry's ``auth.token.jwks`` file holds: this one key, under its thumbprint."""
    return {"keys": [{**public_jwk(public_key), "kid": key_id(public_key), "use": "sig", "alg": "ES256"}]}


def sign_token(
    key: ec.EllipticCurvePrivateKey,
    *,
    issuer: str,
    service: str,
    subject: str,
    access: list[Scope],
    issued_at: int,
    ttl_seconds: int,
) -> str:
    """A Distribution bearer token: an ES256 JWS whose ``access`` claim is the whole grant.

    ``nbf`` is ``iat``, not earlier: the registry already allows a minute of skew either way.
    """
    header = {"alg": "ES256", "typ": "JWT", "kid": key_id(key.public_key())}
    claims = {
        "iss": issuer,
        "sub": subject,
        "aud": service,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": issued_at + ttl_seconds,
        "jti": str(uuid.uuid4()),
        "access": [scope.as_access() for scope in access],
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    r, s = decode_dss_signature(key.sign(signing_input.encode(), ec.ECDSA(hashes.SHA256())))
    # JWS wants r||s, fixed width -- not the DER that `cryptography` returns.
    return f"{signing_input}.{_b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"


# --- The service -----------------------------------------------------------------------------


class Reviewer(Protocol):
    """Who a ServiceAccount token belongs to, for ``audience``; None if nobody."""

    def __call__(self, token: str, audience: str) -> str | None: ...


def _basic_password(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(authorization[len("Basic ") :], validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    _, sep, password = decoded.partition(":")
    return password if sep and password else None


@dataclass(frozen=True, slots=True)
class Reply:
    status: int
    body: dict[str, object]


class TokenServer:
    def __init__(
        self,
        config: TokenServerConfig,
        key: ec.EllipticCurvePrivateKey,
        *,
        review: Reviewer,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._key = key
        self._review = review
        self._clock = clock

    def issue(self, *, authorization: str | None, service: str | None, scopes: list[str]) -> Reply:
        config = self._config
        if service != config.service:
            return Reply(HTTPStatus.BAD_REQUEST, {"errors": [{"code": "UNSUPPORTED", "message": "unknown service"}]})
        token = _basic_password(authorization)
        try:
            subject = self._review(token, config.audience) if token else None
        except Exception:
            logger.exception("token refused: TokenReview failed")
            return Reply(HTTPStatus.SERVICE_UNAVAILABLE, {"errors": [{"code": "UNAVAILABLE"}]})
        if subject is None:
            logger.warning("token refused: the caller presented no ServiceAccount token for %s", config.audience)
            return Reply(HTTPStatus.UNAUTHORIZED, {"errors": [{"code": "UNAUTHORIZED"}]})
        requested = [scope for scope in (parse_scope(s) for s in scopes) if scope is not None]
        granted = grant(subject, requested, config.grants)
        now = int(self._clock())
        issued = sign_token(
            self._key,
            issuer=config.issuer,
            service=config.service,
            subject=subject,
            access=granted,
            issued_at=now,
            ttl_seconds=config.ttl_seconds,
        )
        logger.info(
            "token for %s: asked %s, granted %s",
            sanitize_for_log(subject),
            sanitize_for_log([f"{s.name}:{','.join(s.actions)}" for s in requested]),
            sanitize_for_log([f"{s.name}:{','.join(s.actions)}" for s in granted]),
        )
        issued_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        return Reply(
            HTTPStatus.OK,
            {"token": issued, "access_token": issued, "expires_in": config.ttl_seconds, "issued_at": issued_at},
        )


def _kubernetes_reviewer() -> Reviewer:
    from kubernetes import client

    api = client.AuthenticationV1Api()

    def review(token: str, audience: str) -> str | None:
        body = client.V1TokenReview(spec=client.V1TokenReviewSpec(token=token, audiences=[audience]))
        status = api.create_token_review(body).status
        if not status or not status.authenticated or not status.user:
            return None
        return status.user.username or None

    return review


def _handler(server: TokenServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "nhx-local-token-server"

        def do_GET(self) -> None:  # the name http.server dispatches on
            url = urlsplit(self.path)
            if url.path == "/healthz":
                self._send(Reply(HTTPStatus.OK, {"status": "ok"}))
                return
            if url.path != "/token":
                self._send(Reply(HTTPStatus.NOT_FOUND, {"errors": [{"code": "NOT_FOUND"}]}))
                return
            query = parse_qs(url.query)
            self._send(
                server.issue(
                    authorization=self.headers.get("Authorization"),
                    service=(query.get("service") or [None])[0],
                    scopes=query.get("scope", []),
                )
            )

        def _send(self, reply: Reply) -> None:
            payload = json.dumps(reply.body).encode()
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # `format`: the base class names it so
            return

    return Handler


def load_key(pem: bytes) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError("the token server's key must be a P-256 EC private key")
    return key


def main(argv: list[str]) -> int:
    """``nhx-build token-server`` serves. ``nhx-build token-server jwks`` prints the JWKS for a key on stdin."""
    if argv[:1] == ["jwks"]:
        print(json.dumps(jwks(load_key(sys.stdin.buffer.read()).public_key())))
        return 0
    if argv:
        print("usage: nhx-build token-server [jwks]", file=sys.stderr)
        return 2
    path = os.environ.get("NHX_TOKEN_SERVER_CONFIG")
    if not path:
        logger.error("NHX_TOKEN_SERVER_CONFIG is not set")
        return 2
    from kubernetes import config as k8s_config

    config = TokenServerConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    k8s_config.load_incluster_config()
    server = TokenServer(config, load_key(Path(config.key_file).read_bytes()), review=_kubernetes_reviewer())
    logger.warning("LOCAL-ONLY token service for %s on :%d -- not for any shared cluster", config.service, config.port)
    ThreadingHTTPServer(("", config.port), _handler(server)).serve_forever()
    return 0
