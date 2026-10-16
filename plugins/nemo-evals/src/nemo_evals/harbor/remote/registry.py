# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin a public image tag to its digest with an anonymous registry v2 lookup.

M2 story 0's slice of M3 story 2's registry client: anonymous pulls only, no credentials.
"""

import re
from dataclasses import dataclass

import httpx
from nemo_evals.harbor.remote.profile import PINNED_IMAGE_REF_PATTERN

_DOCKER_HUB = "docker.io"
_DOCKER_HUB_API = "registry-1.docker.io"
_MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_CHALLENGE_PARAM = re.compile(r'(\w+)="([^"]*)"')
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImageDigestUnavailableError(Exception):
    """The tag doesn't resolve anonymously: missing, private, or the registry gave no digest."""


@dataclass(frozen=True)
class ImageName:
    """``registry/repository[:tag]`` split the way Docker reads it."""

    registry: str
    repository: str
    tag: str
    name: str  # the reference as written, without its tag

    @classmethod
    def parse(cls, image: str) -> "ImageName":
        name, _, tag = image.rpartition(":") if ":" in image.rpartition("/")[2] else (image, "", "latest")
        first, _, rest = name.partition("/")
        if rest and ("." in first or ":" in first or first == "localhost"):
            registry, repository = first, rest
        else:
            registry, repository = _DOCKER_HUB, name if rest else f"library/{name}"
        return cls(registry=registry, repository=repository, tag=tag or "latest", name=name)


def is_pinned(image: str) -> bool:
    """True for ``repo@sha256:<64 hex>``."""
    return re.search(PINNED_IMAGE_REF_PATTERN, image) is not None


async def resolve_image_digest(image: str, *, http_client: httpx.AsyncClient | None = None) -> str:
    """Return ``image`` pinned by digest: unchanged when already pinned, else ``<name>@<digest>`` of its tag.

    Raises:
        ImageDigestUnavailableError: The registry refused the anonymous lookup or returned no digest.
    """
    if is_pinned(image):
        return image
    if "@" in image:
        raise ImageDigestUnavailableError(f"{image} names a digest that isn't sha256:<64 hex>")
    ref = ImageName.parse(image)
    host = _DOCKER_HUB_API if ref.registry == _DOCKER_HUB else ref.registry
    url = f"https://{host}/v2/{ref.repository}/manifests/{ref.tag}"
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    try:
        response = await client.head(url, headers={"Accept": _MANIFEST_TYPES})
        if response.status_code == 401:
            token = await _anonymous_token(client, response.headers.get("www-authenticate", ""), ref)
            response = await client.head(url, headers={"Accept": _MANIFEST_TYPES, "Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise ImageDigestUnavailableError(f"{image}: registry lookup failed: {exc}") from exc
    finally:
        if owned:
            await client.aclose()
    if response.status_code != 200:
        raise ImageDigestUnavailableError(f"{image}: registry answered {response.status_code} for tag {ref.tag}")
    digest = response.headers.get("docker-content-digest", "")
    if not _DIGEST.match(digest):
        raise ImageDigestUnavailableError(f"{image}: registry returned no sha256 digest")
    return f"{ref.name}@{digest}"


async def _anonymous_token(client: httpx.AsyncClient, challenge: str, ref: ImageName) -> str:
    scheme, _, params = challenge.partition(" ")
    values = dict(_CHALLENGE_PARAM.findall(params))
    if scheme.lower() != "bearer" or "realm" not in values:
        raise ImageDigestUnavailableError(f"{ref.name}: registry requires credentials ({scheme or 'no challenge'})")
    query = {"scope": values.get("scope") or f"repository:{ref.repository}:pull"}
    if "service" in values:
        query["service"] = values["service"]
    response = await client.get(values["realm"], params=query)
    if response.status_code != 200:
        raise ImageDigestUnavailableError(f"{ref.name}: anonymous token refused ({response.status_code})")
    body = response.json()
    token = body.get("token") or body.get("access_token")
    if not isinstance(token, str) or not token:
        raise ImageDigestUnavailableError(f"{ref.name}: anonymous token response has no token")
    return token
