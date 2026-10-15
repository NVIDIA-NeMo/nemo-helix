# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving an image a build copies: which manifest, by digest, a tag names for a platform.

Read-only and anonymous: it answers a registry's bearer challenge with a pull-scoped token and holds no
credential. The digest is computed from the bytes the registry serves, never taken from a header.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.registry_auth import parse_challenges

MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_INDEX_TYPES = frozenset(
    {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
)
_API_HOSTS = {"docker.io": "registry-1.docker.io"}


class UpstreamError(Exception):
    """The registry couldn't be reached, or answered in a way that isn't accepted."""


class UpstreamNotFound(UpstreamError):
    """The reference doesn't resolve to a public image for the platform."""


@dataclass(frozen=True, slots=True)
class UpstreamManifest:
    """The single-platform manifest a reference names for a platform."""

    reference: ImageReference
    manifest_digest: str


def _platform_matches(candidate: dict, platform: str) -> bool:
    """Whether ``os``/``architecture``/``variant`` is ``os/arch[/variant]``; ``arm64`` with no variant is ``v8``."""
    os_name, _, rest = platform.partition("/")
    architecture, _, variant = rest.partition("/")
    if candidate.get("os") != os_name or candidate.get("architecture") != architecture:
        return False
    if not variant:
        return True
    return (candidate.get("variant") or ("v8" if architecture == "arm64" else "")) == variant


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


class UpstreamRegistry:
    """Resolves public images over HTTPS, answering bearer challenges anonymously."""

    def __init__(self, *, timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
            event_hooks={"request": [_refuse_plain_http]},
        )
        self._tokens: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def resolve(self, reference: ImageReference, *, platform: str) -> UpstreamManifest:
        """The manifest ``reference`` names for ``platform``: the matching child of an index, or a single image
        whose config says it is for ``platform``."""
        manifest = self._get(reference, f"manifests/{reference.reference}", accept=MANIFEST_ACCEPT)
        digest = _sha256(manifest.content)
        stated = manifest.headers.get("Docker-Content-Digest")
        if stated is not None and stated != digest:
            raise UpstreamError(f"{reference} says it is {stated} but hashes to {digest}")
        if reference.digest is not None and digest != reference.digest:
            raise UpstreamError(f"{reference} was served as {digest}")
        document = _json(manifest, str(reference))
        if manifest.headers.get("Content-Type", "").split(";")[0] in _INDEX_TYPES:
            for entry in document.get("manifests") or []:
                if isinstance(entry, dict) and _platform_matches(entry.get("platform") or {}, platform):
                    return UpstreamManifest(reference, str(entry["digest"]))
            raise UpstreamNotFound(f"{reference} has no {platform} image")
        config_digest = (document.get("config") or {}).get("digest")
        if not isinstance(config_digest, str):
            raise UpstreamError(f"{reference} names no config")
        config = self._get(reference, f"blobs/{config_digest}", accept="*/*")
        if _sha256(config.content) != config_digest:
            raise UpstreamError(f"{reference} served its config with the wrong content")
        if not _platform_matches(_json(config, str(reference)), platform):
            raise UpstreamNotFound(f"{reference} is a single image that isn't for {platform}")
        return UpstreamManifest(reference, digest)

    def _get(self, reference: ImageReference, path: str, *, accept: str) -> httpx.Response:
        repository = f"{reference.registry}/{reference.repository}"
        url = f"https://{_API_HOSTS.get(reference.registry, reference.registry)}/v2/{reference.repository}/{path}"
        headers = {"Accept": accept}
        try:
            if token := self._tokens.get(repository):
                headers["Authorization"] = token
            response = self._client.get(url, headers=headers)
            if response.status_code == 401:
                self._tokens.pop(repository, None)
                token = self._token(reference, response.headers.get("WWW-Authenticate", ""))
                if token:
                    self._tokens[repository] = headers["Authorization"] = token
                    response = self._client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"{reference.registry}: {type(exc).__name__}: {exc}") from exc
        if response.status_code in (401, 403, 404):
            raise UpstreamNotFound(f"{reference} doesn't exist, or isn't public")
        if response.status_code != 200:
            raise UpstreamError(f"{reference} {path} returned {response.status_code}")
        return response

    def _token(self, reference: ImageReference, header: str) -> str | None:
        """A pull-scoped bearer token from the challenge's realm, which must be HTTPS."""
        bearer = dict(parse_challenges(header)).get("bearer")
        realm = (bearer or {}).get("realm")
        if not bearer or not realm:
            return None
        if urlsplit(realm).scheme != "https":
            raise UpstreamError(f"{reference.registry} named a token endpoint that isn't HTTPS")
        params = {
            "service": bearer.get("service", reference.registry),
            "scope": f"repository:{reference.repository}:pull",
        }
        response = self._client.get(realm, params=params)
        if response.status_code != 200:
            raise UpstreamError(f"{reference.registry}'s token endpoint returned {response.status_code}")
        payload = _json(response, f"{reference.registry} token")
        token = payload.get("token") or payload.get("access_token")
        return f"Bearer {token}" if isinstance(token, str) and token else None


def _json(response: httpx.Response, what: str) -> dict:
    try:
        document = response.json()
    except ValueError as exc:
        raise UpstreamError(f"{what} isn't JSON") from exc
    if not isinstance(document, dict):
        raise UpstreamError(f"{what} isn't a JSON object")
    return document


def _refuse_plain_http(request: httpx.Request) -> None:
    if request.url.scheme != "https":
        raise UpstreamError(f"refusing a plain-HTTP request to {request.url.host}")
