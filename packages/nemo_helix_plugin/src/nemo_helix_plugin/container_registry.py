# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image references and auth challenges, parsed the same way by every OCI registry client in the platform."""

from __future__ import annotations

import re
from dataclasses import dataclass

DOCKER_HUB_REGISTRY = "registry-1.docker.io"
_DOCKER_HUB_ALIASES = frozenset({"docker.io", "index.docker.io", DOCKER_HUB_REGISTRY})

_REGISTRY = re.compile(r"(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?|\[[0-9a-f:.]+\])(?::[0-9]{1,5})?")
_REPOSITORY_SEGMENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*")
_TAG = re.compile(r"[\w][\w.-]{0,127}")
_DIGEST = re.compile(r"[a-z0-9]+(?:[.+_-][a-z0-9]+)*:[a-zA-Z0-9=_-]{32,}")

# One `WWW-Authenticate` challenge's scheme, then its parameters: httpx joins repeated headers with `, `.
_SCHEME = re.compile(r"[\s,]*([A-Za-z][A-Za-z0-9!#$%&'*+.^_`|~-]*)(?=\s|$)")
_PARAM = re.compile(r'\s*([A-Za-z0-9!#$%&\'*+.^_`|~-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^\s,]*)\s*(?:,|$)')


class ImageReferenceError(ValueError):
    """The text is not an image reference."""


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str

    @property
    def is_digest(self) -> bool:
        return ":" in self.reference

    def __str__(self) -> str:
        separator = "@" if self.is_digest else ":"
        return f"{self.registry}/{self.repository}{separator}{self.reference}"


def normalize_registry(registry: str) -> str:
    """Lowercase, drop the default https port, and collapse Docker Hub's aliases."""
    normalized = registry.strip().lower().removesuffix(":443")
    return DOCKER_HUB_REGISTRY if normalized in _DOCKER_HUB_ALIASES else normalized


def parse_image_reference(image: str) -> ImageReference:
    """Split ``[registry/]repository[:tag][@digest]`` the way ``docker pull`` does."""
    text = image.strip()
    invalid = ImageReferenceError(f"'{image}' is not an image reference. Use registry/repository:tag.")
    if not text or "://" in text or any(char.isspace() for char in text):
        raise invalid

    name, _, digest = text.partition("@")
    last_slash = name.rfind("/")
    colon = name.rfind(":")
    tag = ""
    if colon > last_slash:
        name, tag = name[:colon], name[colon + 1 :]

    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = normalize_registry(first), rest
    else:
        registry, repository = DOCKER_HUB_REGISTRY, name
    if registry == DOCKER_HUB_REGISTRY and "/" not in repository:
        repository = f"library/{repository}"

    if (
        not _REGISTRY.fullmatch(registry)
        or not all(_REPOSITORY_SEGMENT.fullmatch(segment) for segment in repository.split("/"))
        or (tag and not _TAG.fullmatch(tag))
        or (digest and not _DIGEST.fullmatch(digest))
        or ("@" in text and not digest)
        or (colon > last_slash and not tag)
    ):
        raise invalid
    return ImageReference(registry=registry, repository=repository, reference=digest or tag or "latest")


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
