# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every OCI registry client in the platform parses the same way: image references and auth challenges."""

from __future__ import annotations

import re
from dataclasses import dataclass

DOCKER_HUB_REGISTRY = "registry-1.docker.io"
_DOCKER_HUB_ALIASES = frozenset({"docker.io", "index.docker.io", DOCKER_HUB_REGISTRY})

#: One `WWW-Authenticate` challenge's scheme, then each of its parameters, read per challenge: a
#: header can carry several, and httpx joins repeated headers with `, `.
_SCHEME = re.compile(r"[\s,]*([A-Za-z][A-Za-z0-9!#$%&'*+.^_`|~-]*)(?=\s|$)")
_PARAM = re.compile(r'\s*([A-Za-z0-9!#$%&\'*+.^_`|~-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^\s,]*)\s*(?:,|$)')


class ImageReferenceError(ValueError):
    """The text is not an image reference."""


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str

    def __str__(self) -> str:
        return f"{self.registry}/{self.repository}:{self.reference}"


def parse_image_reference(image: str) -> ImageReference:
    """Split ``[registry/]repository[:tag][@digest]`` the way ``docker pull`` does.

    The registry is lowercased and Docker Hub's aliases collapse to its registry host,
    with single-segment repositories placed under ``library/``.
    """
    text = image.strip()
    if not text or "://" in text or any(char.isspace() for char in text):
        raise ImageReferenceError(f"'{image}' is not an image reference. Use registry/repository:tag.")

    name, _, digest = text.partition("@")
    last_slash = name.rfind("/")
    colon = name.rfind(":")
    tag = ""
    if colon > last_slash:
        name, tag = name[:colon], name[colon + 1 :]

    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first.lower(), rest
    else:
        registry, repository = DOCKER_HUB_REGISTRY, name

    if registry in _DOCKER_HUB_ALIASES:
        registry = DOCKER_HUB_REGISTRY
        if "/" not in repository:
            repository = f"library/{repository}"

    if not repository or "" in repository.split("/"):
        raise ImageReferenceError(f"'{image}' is not an image reference. Use registry/repository:tag.")
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
