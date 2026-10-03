# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OCI distribution calls the push step makes itself to store a signature: read a manifest, write blobs and manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any

import httpx
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
INDEX_MEDIA_TYPES = frozenset({OCI_INDEX, DOCKER_MANIFEST_LIST})

#: Per request. Artifactory has been seen taking two minutes to accept a manifest.
REGISTRY_TIMEOUT_SECONDS = 300

#: crane's own policy, which pushes the image: each request is tried three times, a second and then three apart,
#: if the connection fails or times out, or the registry answers with one of these.
ATTEMPTS = 3
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def registry_client(timeout: float) -> httpx.Client:
    """A client for a registry and its token service, which follows no redirect and keeps no cookie.

    A redirect would carry a credential somewhere the registry never named. A cookie sent back changes
    what a request may do: Harbor checks CSRF on any request carrying its session, and refuses the write.
    """
    jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    return httpx.Client(timeout=timeout, follow_redirects=False, cookies=jar)


class RegistryError(Exception):
    """The registry failed or refused a call."""


class ManifestNotFound(RegistryError):
    """Nothing is at that tag or digest."""


def digest_of(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class Registry:
    """One registry's manifests and blobs, reached with one `Authorization` header (see `registry_auth.login`)."""

    def __init__(
        self,
        host: str,
        authorization: str | None,
        *,
        http: httpx.Client,
        plain_http: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._origin = httpx.URL(f"{'http' if plain_http else 'https'}://{host}")
        self._authorization = authorization
        self._http = http
        self._sleep = sleep

    def manifest(self, repository: str, reference: str) -> tuple[str, dict[str, Any]]:
        """The manifest at ``reference``, and its digest, computed from the bytes the registry returned."""
        response = self._send(
            "GET",
            self._origin.join(f"/v2/{repository}/manifests/{reference}"),
            headers={"Accept": ", ".join((OCI_MANIFEST, OCI_INDEX, DOCKER_MANIFEST, DOCKER_MANIFEST_LIST))},
        )
        if response.status_code == 404:
            raise ManifestNotFound(f"nothing is at {repository}:{reference}")
        self._expect(response, 200, f"reading {repository}:{reference}")
        try:
            manifest = json.loads(response.content)
        except ValueError as exc:
            raise RegistryError(f"{repository}:{reference} is not a JSON manifest") from exc
        if not isinstance(manifest, dict):
            raise RegistryError(f"{repository}:{reference} is not a JSON manifest")
        return digest_of(response.content), manifest

    def put_blob(self, repository: str, data: bytes) -> str:
        """Upload ``data`` in one request, and return its digest."""
        digest = digest_of(data)
        started = self._send("POST", self._origin.join(f"/v2/{repository}/blobs/uploads/"))
        self._expect(started, 202, f"starting an upload to {repository}")
        location = self._origin.join(started.headers.get("Location", ""))
        # The credential goes only to the registry it is for.
        if (location.scheme, location.host, location.port) != (
            self._origin.scheme,
            self._origin.host,
            self._origin.port,
        ):
            raise RegistryError(f"the registry named an upload location on another host: {location.host}")
        done = self._send(
            "PUT",
            location.copy_merge_params({"digest": digest}),
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        self._expect(done, 201, f"uploading a blob to {repository}")
        return digest

    def put_manifest(self, repository: str, reference: str, manifest: bytes, media_type: str) -> None:
        response = self._send(
            "PUT",
            self._origin.join(f"/v2/{repository}/manifests/{reference}"),
            content=manifest,
            headers={"Content-Type": media_type},
        )
        self._expect(response, 201, f"writing {repository}:{reference}")

    def _send(
        self, method: str, url: httpx.URL, *, headers: dict[str, str] | None = None, content: bytes | None = None
    ) -> httpx.Response:
        auth = {"Authorization": self._authorization} if self._authorization else {}
        attempt = 1
        while True:
            last = attempt == ATTEMPTS
            try:
                response = self._http.request(method, url, headers={**auth, **(headers or {})}, content=content)
            except httpx.HTTPError as exc:
                if last or not isinstance(exc, _TRANSIENT):
                    raise RegistryError(f"{self._origin.host} could not be reached: {type(exc).__name__}") from exc
                failure = type(exc).__name__
            else:
                if last or response.status_code not in RETRY_STATUSES:
                    return response
                failure = f"answered {response.status_code}"
            logger.warning("%s %s: %s; retrying", method, sanitize_for_log(url.path), failure)
            self._sleep(3.0 ** (attempt - 1))
            attempt += 1

    @staticmethod
    def _expect(response: httpx.Response, status: int, doing: str) -> None:
        if response.status_code != status:
            # The body is not echoed: a registry's error page is not ours to repeat.
            raise RegistryError(f"{doing}: the registry answered {response.status_code}")
