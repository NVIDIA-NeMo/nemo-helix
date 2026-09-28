# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the reconciler speaks to a registry.

One protocol, and it is not vendor-specific: GAR, ECR, ACR, Harbor, Quay and the ``distribution``
reference implementation all implement the **OCI Distribution Specification**. So this is a
client for that spec rather than for any registry.

**Why the registry at all, rather than the build's own report.** The build can write a digest
file -- kaniko does, with ``--digest-file`` -- but it writes it from the sandbox, the one pod that
runs caller-authored ``RUN``. Trusting a reported digest would mean trusting untrusted code about
the identity of what it produced, which is the one thing image identity cannot take on faith. The
registry is the one source the build cannot author.

Two notes that cost time to rediscover:

- ``HEAD /v2/`` is **not** a supported method. A ``405`` there is correct, not broken; the
  handshake is a ``GET``.
- The auth flow is a challenge: an unauthenticated request returns ``401`` with a
  ``WWW-Authenticate`` header, in one of two schemes. ``Bearer realm=...,service=...,scope=...``
  names a *token endpoint*, where you exchange a basic credential for a scoped bearer token --
  Docker Hub, GAR, ECR, Harbor. ``Basic realm="..."`` -- ``distribution`` with htpasswd -- wants
  the credential itself, and its realm is a label, not a URL. Sending basic auth unprompted works
  on some registries and not on others, so this client answers whichever challenge it is given.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

#: Ask for every manifest media type, so an index and a plain manifest both resolve.
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

_INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: One `WWW-Authenticate` challenge's scheme, then each of its `name=value` parameters. A header
#: can carry several challenges -- `Bearer ..., Basic ...` -- and httpx joins repeated headers
#: with `, `, so parameters are read per challenge rather than across the whole header.
_SCHEME = re.compile(r"[\s,]*([A-Za-z][A-Za-z0-9!#$%&'*+.^_`|~-]*)(?=\s|$)")
_PARAM = re.compile(r'\s*([A-Za-z0-9!#$%&\'*+.^_`|~-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^\s,]*)\s*(?:,|$)')

#: Registries whose API is not served at the hostname references use. Docker Hub's is the one
#: that matters: `docker.io` answers `/v2/` with a redirect to its website.
_API_HOSTS = {"docker.io": "registry-1.docker.io", "index.docker.io": "registry-1.docker.io"}


def _api_host(registry: str) -> str:
    return _API_HOSTS.get(registry, registry)


def parse_challenges(header: str) -> list[tuple[str, dict[str, str]]]:
    """``[(scheme, {param: value})]``, schemes and parameter names lowercased (RFC 9110)."""
    challenges: list[tuple[str, dict[str, str]]] = []
    position = 0
    while scheme := _SCHEME.match(header, position):
        position = scheme.end()
        params: dict[str, str] = {}
        while param := _PARAM.match(header, position):
            value = param.group(2)
            if value.startswith('"'):
                value = re.sub(r"\\(.)", r"\1", value[1:-1])
            params[param.group(1).lower()] = value
            position = param.end()
        challenges.append((scheme.group(1).lower(), params))
    return challenges


class RegistryError(Exception):
    """The registry could not be reached, or answered in a way we do not accept."""


class ReferenceNotFound(RegistryError):
    """The reference does not resolve. Distinct from an error, because it may simply be early."""


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """What one registry pass observed."""

    digest: str
    manifest_digest: str
    media_type: str


def signature_tag(digest: str) -> str:
    """The legacy cosign signature tag for a digest: ``sha256-<hex>.sig``.

    A cosign signature lives either here or as an OCI 1.1 referrer, and **cosign v3 defaults to
    referrers**. A verifier that reads only this tag then finds nothing and reports *"unsigned
    image"* rather than *"format mismatch"* -- which is why the storage mode is pinned to the
    oldest verifier a deployment serves, and why this function exists rather than being inlined.
    """
    return digest.replace(":", "-") + ".sig"


class RegistryClient:
    """A small OCI Distribution client, scoped to what the reconciler needs.

    Deliberately read-only: it resolves references and checks whether an artifact exists. It
    cannot push, and it holds a credential that does not need to.
    """

    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
        insecure: bool = False,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._username = username
        self._password = password
        # `http` for a dev registry only. A registry reached over plain HTTP offers no assurance
        # that the digest it reports is the digest anyone else would see, which is the one thing
        # this client exists to establish -- so it is opt-in, off by default, and never a
        # fallback after an HTTPS attempt fails.
        self._scheme = "http" if insecure else "https"
        # `transport` is a test seam: an httpx.MockTransport stands in for a registry. Redirects
        # are followed, and httpx drops `Authorization` when one crosses to another host.
        self._client = httpx.Client(timeout=timeout, follow_redirects=True, transport=transport)
        #: The `Authorization` value that last answered each repository's challenge -- a bearer
        #: token or the basic credential, whichever the registry asked for.
        self._authorization: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RegistryClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- auth ---------------------------------------------------------------

    def _exchange_token(self, registry: str, repository: str, params: dict[str, str]) -> str | None:
        """Exchange the basic credential for a fresh, scoped bearer token, per the challenge.

        Always a fresh exchange -- caching is `_get`'s job, because only `_get` sees the 401 that
        says a cached token has gone stale.

        **The realm is the registry's to name, so it is checked before anything is sent to it.**
        It must be HTTPS unless this client is talking to a registry over HTTP anyway: the
        credential goes to it, and a realm of ``http://169.254.169.254/...`` would otherwise
        receive it in clear. It is not echoed in errors, which surface in API responses.
        """
        realm = params.get("realm")
        if not realm:
            return None
        if self._scheme == "https" and urlsplit(realm).scheme != "https":
            raise RegistryError(f"{registry} named a token endpoint that is not HTTPS")

        query = {"service": params.get("service", registry), "scope": f"repository:{repository}:pull"}
        auth = (self._username, self._password) if self._username and self._password else None
        response = self._client.get(realm, params=query, auth=auth)
        if response.status_code != 200:
            raise RegistryError(f"{registry}'s token endpoint returned {response.status_code}")

        payload = response.json()
        token = payload.get("token") or payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise RegistryError(f"{registry}'s token endpoint returned no token")
        return token

    def _answer_challenge(self, registry: str, repository: str, header: str) -> str | None:
        """The `Authorization` value that answers a 401's challenge, or None if nothing can.

        **Both schemes, because registries use both.** An earlier version handled only Bearer,
        and read a Basic challenge's realm -- ``"Registry Realm"``, a label -- as a token URL. The
        request failed with an httpx error that is not a :class:`RegistryError`, so it escaped the
        reconcile loop's handling, never counted against the attempt budget, and left every row
        on such a registry ``pending`` forever. Found against ``distribution`` with htpasswd.

        The credential goes only to a registry that asked for it, and only to the one a row
        records -- which is the deployment's own, since a caller cannot name a registry.

        **Bearer first, when a registry offers both.** A header can carry several challenges,
        and reading parameters across all of them let a later ``Basic realm="..."`` label
        overwrite the Bearer realm -- which was then fetched as a URL.
        """
        challenges = dict(reversed(parse_challenges(header)))
        if "bearer" in challenges:
            token = self._exchange_token(registry, repository, challenges["bearer"])
            return f"Bearer {token}" if token else None
        if "basic" in challenges and self._username and self._password:
            return "Basic " + base64.b64encode(f"{self._username}:{self._password}".encode()).decode()
        return None

    def _get(self, registry: str, repository: str, path: str, *, accept: str) -> httpx.Response:
        """GET, answering an auth challenge once; every failure to get an answer a RegistryError.

        A transport failure, a timeout, or a token endpoint answering something other than JSON
        is a :class:`RegistryError` like any other. Escaping as anything else, it would bypass
        the reconciler's handling and be retried forever with a traceback each time.
        """
        try:
            return self._get_answering_challenge(registry, repository, path, accept=accept)
        except httpx.HTTPError as exc:
            raise RegistryError(f"{registry}: {type(exc).__name__}: {exc}") from exc
        except ValueError as exc:  # `response.json()` on a body that is not JSON
            raise RegistryError(f"{registry} answered with something that is not JSON") from exc

    def _get_answering_challenge(self, registry: str, repository: str, path: str, *, accept: str) -> httpx.Response:
        """GET, answering an auth challenge once and remembering the answer.

        **A 401 invalidates the cached answer.** An earlier version cached one bearer token per
        repository for the life of the process and never dropped it. Bearer tokens are
        short-lived, so once one expired every lookup presented it, the single retry 401'd again,
        and the reconciler could no longer resolve anything -- against a real registry, within
        the hour. Now a 401 drops the cache entry and triggers exactly one fresh answer; a second
        401 is returned to the caller as a real authorization failure rather than retried.

        The cached answer is also sent on the FIRST request, which removes the unauthenticated
        round trip every call used to make just to be told to authenticate.
        """
        url = f"{self._scheme}://{_api_host(registry)}/v2/{repository}/{path}"
        cache_key = f"{registry}/{repository}"
        headers = {"Accept": accept}

        cached = self._authorization.get(cache_key)
        if cached:
            headers["Authorization"] = cached
        response = self._client.get(url, headers=headers)
        if response.status_code != 401:
            return response

        # Stale, revoked, or never had one. Either way the cached value is no longer trusted.
        self._authorization.pop(cache_key, None)
        authorization = self._answer_challenge(registry, repository, response.headers.get("WWW-Authenticate", ""))
        if not authorization:
            return response
        self._authorization[cache_key] = authorization
        headers["Authorization"] = authorization
        return self._client.get(url, headers=headers)

    # -- reads --------------------------------------------------------------

    def ping(self, registry: str) -> bool:
        """The ``GET /v2/`` handshake. A 401 counts as reachable -- it means the API is there."""
        response = self._client.get(f"{self._scheme}://{_api_host(registry)}/v2/")
        return response.status_code in (200, 401)

    def resolve(self, registry: str, repository: str, reference: str) -> ResolvedImage:
        """Resolve a tag (or digest) to the two digests that name the image.

        ``digest`` is whatever the reference resolved to. If that document is an **index**, it
        descends exactly one level for ``manifest_digest``; if it is already a plain manifest --
        the usual case under this execution mode -- the two are equal and there is nothing to
        descend.

        **The digest is computed from the bytes served, not read off a header.** A content
        address is only worth something if it addresses the content: taking the registry's
        ``Docker-Content-Digest`` at its word would let the body -- and the index child chosen
        from it -- be anything. The header, where a registry sends one, must agree.
        """
        response = self._get(registry, repository, f"manifests/{reference}", accept=MANIFEST_ACCEPT)
        if response.status_code == 404:
            raise ReferenceNotFound(f"{registry}/{repository}:{reference} does not resolve")
        if response.status_code != 200:
            raise RegistryError(f"{registry}/{repository}:{reference} returned {response.status_code}")

        digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
        stated = response.headers.get("Docker-Content-Digest")
        if stated is not None and stated != digest:
            raise RegistryError(f"{registry}/{repository}:{reference} says it is {stated} but hashes to {digest}")
        if _DIGEST.fullmatch(reference) and digest != reference:
            # Asked for a digest, served something else. Content-addressed storage makes this
            # impossible for an honest registry, so it is refused rather than recorded.
            raise RegistryError(f"{registry}/{repository}@{reference} was served as {digest}")

        media_type = response.headers.get("Content-Type", "").split(";")[0]
        manifest_digest = digest
        if media_type in _INDEX_TYPES:
            manifests = self._index_entries(response, digest)
            manifest_digest = manifests[0]["digest"]

        return ResolvedImage(digest=digest, manifest_digest=manifest_digest, media_type=media_type)

    @staticmethod
    def _index_entries(response: httpx.Response, digest: str) -> list[dict]:
        """An index's entries, each with a well-formed digest, or a RegistryError."""
        try:
            document = response.json()
        except ValueError as exc:
            raise RegistryError(f"index {digest} is not JSON") from exc
        manifests = document.get("manifests") if isinstance(document, dict) else None
        if not isinstance(manifests, list) or not manifests:
            raise RegistryError(f"index {digest} lists no manifests")
        for entry in manifests:
            if not isinstance(entry, dict) or not _DIGEST.fullmatch(str(entry.get("digest", ""))):
                raise RegistryError(f"index {digest} lists an entry without a sha256 digest")
        return manifests

    def exists(self, registry: str, repository: str, reference: str) -> bool:
        """Whether a reference resolves at all. Used for the signature presence check.

        **Presence, not validity.** This says an artifact is *there*, not that a trusted key made
        it -- anything able to write to the repository can place a well-formed signature made
        with any key. The cluster's own verifier is what stands between a bad signature and a
        running pod. What this adds is a pipeline correctness check: it catches signing having
        silently not happened, and catching that needs no public key.
        """
        try:
            self.resolve(registry, repository, reference)
        except ReferenceNotFound:
            return False
        return True
