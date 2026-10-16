# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving an image a build copies: which manifest, by digest, a tag names for a platform.

Read-only and anonymous: it answers a registry's bearer challenge with a pull-scoped token and holds no
credential. The digest is computed from the bytes the registry serves, never taken from a header.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from nemo_builder_plugin.identity import DIGEST_PATTERN, ImageReference
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


def _platform_name(candidate: dict) -> str:
    os_name, architecture, variant = candidate.get("os"), candidate.get("architecture"), candidate.get("variant")
    if not variant or (architecture == "arm64" and variant == "v8"):
        return f"{os_name}/{architecture}"
    return f"{os_name}/{architecture}/{variant}"


def _platform_entries(document: dict) -> list[dict]:
    """An index's entries that name a platform."""
    return [
        entry
        for entry in document.get("manifests") or []
        if isinstance(entry, dict) and isinstance(entry.get("platform"), dict)
    ]


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
        digest, document, is_index = self._manifest(reference)
        if is_index:
            for entry in _platform_entries(document):
                if _platform_matches(entry["platform"], platform):
                    child = entry.get("digest")
                    if not isinstance(child, str) or not re.fullmatch(DIGEST_PATTERN, child):
                        raise UpstreamError(f"{reference} lists its {platform} image without a sha256 digest")
                    return UpstreamManifest(reference, child)
            raise UpstreamNotFound(f"{reference} has no {platform} image")
        if not _platform_matches(self._config(reference, document), platform):
            raise UpstreamNotFound(f"{reference} is a single image that isn't for {platform}")
        return UpstreamManifest(reference, digest)

    def platforms(self, reference: ImageReference) -> list[str]:
        """The platforms ``reference`` has an image for, as ``os/architecture[/variant]``, sorted.

        An index's entries for no real platform, such as the attestations BuildKit adds as ``unknown/unknown``,
        are left out. ``arm64`` is named without its default variant, ``v8``.
        """
        _, document, is_index = self._manifest(reference)
        if is_index:
            names = {_platform_name(entry["platform"]) for entry in _platform_entries(document)}
            return sorted(name for name in names if not name.startswith("unknown/"))
        return [_platform_name(self._config(reference, document))]

    def _manifest(self, reference: ImageReference) -> tuple[str, dict, bool]:
        """The digest and document ``reference`` names, and whether it is an index."""
        manifest = self._get(reference, f"manifests/{reference.reference}", accept=MANIFEST_ACCEPT)
        digest = _sha256(manifest.content)
        stated = manifest.headers.get("Docker-Content-Digest")
        if stated is not None and stated != digest:
            raise UpstreamError(f"{reference} says it is {stated} but hashes to {digest}")
        if reference.digest is not None and digest != reference.digest:
            raise UpstreamError(f"{reference} was served as {digest}")
        is_index = manifest.headers.get("Content-Type", "").split(";")[0] in _INDEX_TYPES
        return digest, _json(manifest, str(reference)), is_index

    def _config(self, reference: ImageReference, document: dict) -> dict:
        """A single image's config, which says its platform."""
        config_digest = (document.get("config") or {}).get("digest")
        if not isinstance(config_digest, str):
            raise UpstreamError(f"{reference} names no config")
        config = self._get(reference, f"blobs/{config_digest}", accept="*/*")
        if _sha256(config.content) != config_digest:
            raise UpstreamError(f"{reference} served its config with the wrong content")
        return _json(config, str(reference))

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
