# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container image reference parsing and syntax validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Lowercase hex only, so that one digest has one spelling.
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_REGISTRY = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?")
_REPOSITORY_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*")
# ASCII only: Python's `\w` would also admit Unicode letters, which a registry refuses.
TAG_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"


class ImageIdentityError(ValueError):
    """An invalid registry host, repository path or tag."""


def validate_repository(repository: str) -> str:
    """A repository path: ``/``-separated components, none empty and none ``..``."""
    if not repository or any(not _REPOSITORY_COMPONENT.fullmatch(part) for part in repository.split("/")):
        raise ImageIdentityError(f"invalid repository path: {repository!r}")
    return repository


def validate_tag(tag: str) -> str:
    if not re.fullmatch(TAG_PATTERN, tag):
        raise ImageIdentityError(f"invalid tag: {tag!r}")
    return tag


def validate_registry_host(registry: str) -> str:
    host = registry.strip().lower()
    if not _REGISTRY.fullmatch(host):
        raise ImageIdentityError(f"invalid registry host: {registry!r}")
    return host


_DOCKER_HUB = "docker.io"
_DOCKER_HUB_ALIASES = frozenset({"index.docker.io", "registry-1.docker.io", "registry.hub.docker.com"})


@dataclass(frozen=True, slots=True)
class ImageReference:
    """A fully qualified image reference: a registry, a repository, and a tag or a digest."""

    registry: str
    repository: str
    tag: str | None = None
    digest: str | None = None

    @property
    def reference(self) -> str:
        """The tag or digest, as a registry's manifests endpoint takes it."""
        return self.digest or self.tag or ""

    def __str__(self) -> str:
        return f"{self.registry}/{self.repository}" + (f"@{self.digest}" if self.digest else f":{self.tag}")


def normalize_registry(registry: str) -> str:
    """A registry host, with Docker Hub's other hostnames folded into ``docker.io``."""
    host = validate_registry_host(registry)
    return _DOCKER_HUB if host in _DOCKER_HUB_ALIASES else host


def normalize_reference(image_ref: str) -> ImageReference:
    """Expand Docker shorthand and parse: ``alpine`` is ``docker.io/library/alpine:latest``.

    A reference with both a tag and a digest keeps the digest, which is what identifies the image.
    """
    value = image_ref.strip()
    if not value or any(char.isspace() for char in value) or "://" in value:
        raise ImageIdentityError(f"invalid image reference: {image_ref!r}")
    name, at, digest = value.partition("@")
    if at and not re.fullmatch(DIGEST_PATTERN, digest):
        raise ImageIdentityError(f"invalid digest in image reference: {image_ref!r}")
    first, slash, rest = name.partition("/")
    if slash and ("." in first or ":" in first or first.lower() == "localhost"):
        registry, path = normalize_registry(first), rest
    else:
        registry, path = _DOCKER_HUB, name
    if registry == _DOCKER_HUB and "/" not in path:
        path = f"library/{path}"
    tag: str | None = None
    repository, colon, last_tag = path.rpartition(":")
    if colon and "/" not in last_tag:
        path, tag = repository, validate_tag(last_tag)
    if at:
        tag = None
    elif tag is None:
        tag = "latest"
    return ImageReference(registry, validate_repository(path), tag=tag, digest=digest or None)
