# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, repository paths and tags, and the system tag."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

#: Lowercase hex only, so that one digest has one spelling.
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
#: An OCI platform as builds and pulls name it: ``os/architecture``, with an optional ``/variant``.
PLATFORM_PATTERN = r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$"

_REGISTRY = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?")
_REPOSITORY_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*")
# ASCII only: Python's `\w` would also admit Unicode letters, which a registry refuses.
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


class ImageIdentityError(ValueError):
    """An invalid registry host, repository path or tag."""


def validate_repository(repository: str) -> str:
    """A repository path: ``/``-separated components, none empty and none ``..``."""
    if not repository or any(not _REPOSITORY_COMPONENT.fullmatch(part) for part in repository.split("/")):
        raise ImageIdentityError(f"invalid repository path: {repository!r}")
    return repository


def validate_tag(tag: str) -> str:
    if not _TAG.fullmatch(tag):
        raise ImageIdentityError(f"invalid tag: {tag!r}")
    return tag


def validate_caller_tag(tag: str) -> str:
    """A tag a caller may push: valid, and shaped like neither a system tag nor cosign's ``sha256-<hex>.sig``.

    Pushing either would replace the build's own tag for an image, or the image's signature.
    """
    validate_tag(tag)
    if "--" in tag:
        raise ImageIdentityError(f"tag {tag!r} contains '--', which is reserved for system tags")
    if tag.startswith("sha256-"):
        raise ImageIdentityError(f"tag {tag!r} starts with 'sha256-', which is reserved for signatures")
    return tag


def validate_registry_host(registry: str) -> str:
    host = registry.strip().lower()
    if not _REGISTRY.fullmatch(host):
        raise ImageIdentityError(f"invalid registry host: {registry!r}")
    return host


def compose_system_tag(workspace: str, image: str) -> str:
    """``<workspace>--<image>``: the tag a build pushes for one image.

    Caller tags can't contain ``--``, so no caller can push one.
    """
    if "--" in workspace or "--" in image:
        raise ImageIdentityError(f"workspace and image must not contain '--', got {workspace!r} and {image!r}")
    tag = f"{workspace}--{image}"
    if not _TAG.fullmatch(tag):
        raise ImageIdentityError(f"composed system tag is not a valid tag: {tag!r}")
    return tag


# BuildKit's other spellings of an architecture, and the name OCI gives it.
_ARCHITECTURE_ALIASES = {"x86_64": "amd64", "x86-64": "amd64", "aarch64": "arm64"}
_PLATFORM_PART = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Platform:
    """An OCI platform: an os and an architecture, with an optional variant.

    Named as BuildKit names it: in lowercase, with ``x86_64`` and ``x86-64`` as ``amd64``, ``aarch64`` as ``arm64``,
    and arm64's default variant, ``v8``, left out. So two spellings of one platform are equal.
    """

    os: str
    architecture: str
    variant: str | None = None

    @classmethod
    def parse(cls, value: str) -> Platform:
        """The platform ``os/architecture[/variant]`` names, in any spelling BuildKit takes."""
        parts = value.lower().split("/")
        if len(parts) not in (2, 3):
            raise ImageIdentityError(f"invalid platform: {value!r}; expected os/architecture[/variant]")
        return cls._named(parts[0], parts[1], parts[2] if len(parts) == 3 else None, spelled=value)

    @classmethod
    def from_oci(cls, platform: Mapping[str, object]) -> Platform:
        """The platform an image config or an index entry names in its ``os``, ``architecture`` and ``variant``."""
        os_name, architecture, variant = platform.get("os"), platform.get("architecture"), platform.get("variant")
        if not isinstance(os_name, str) or not isinstance(architecture, str) or not isinstance(variant, str | None):
            raise ImageIdentityError(f"invalid platform: {dict(platform)!r}")
        return cls._named(
            os_name.lower(), architecture.lower(), variant.lower() if variant else None, spelled=str(dict(platform))
        )

    @classmethod
    def _named(cls, os_name: str, architecture: str, variant: str | None, *, spelled: str) -> Platform:
        architecture = _ARCHITECTURE_ALIASES.get(architecture, architecture)
        if architecture == "arm64" and variant == "v8":
            variant = None
        if not all(_PLATFORM_PART.fullmatch(part) for part in (os_name, architecture, *([variant] if variant else []))):
            raise ImageIdentityError(f"invalid platform: {spelled!r}; expected os/architecture[/variant]")
        return cls(os_name, architecture, variant or None)

    def __str__(self) -> str:
        return f"{self.os}/{self.architecture}" + (f"/{self.variant}" if self.variant else "")


def platform_name(value: object) -> object:
    """``value`` as BuildKit names that platform, for a model's field validator; anything unreadable is left for the
    field's own pattern to refuse."""
    if not isinstance(value, str):
        return value
    try:
        return str(Platform.parse(value))
    except ImageIdentityError:
        return value


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
