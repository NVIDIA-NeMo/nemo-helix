# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, repository paths and tags, and the system tag."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from nemo_helix_plugin.images import (
    DIGEST_PATTERN as DIGEST_PATTERN,
)
from nemo_helix_plugin.images import (
    TAG_PATTERN,
    validate_tag,
)
from nemo_helix_plugin.images import (
    ImageIdentityError as ImageIdentityError,
)
from nemo_helix_plugin.images import (
    ImageReference as ImageReference,
)
from nemo_helix_plugin.images import (
    normalize_reference as normalize_reference,
)
from nemo_helix_plugin.images import (
    normalize_registry as normalize_registry,
)
from nemo_helix_plugin.images import (
    validate_registry_host as validate_registry_host,
)
from nemo_helix_plugin.images import (
    validate_repository as validate_repository,
)

#: A platform as a request may spell it, ``os/architecture`` with an optional ``/variant``: in any case, and with
#: BuildKit's aliases such as ``x86_64``. Requests name it as :class:`Platform` does before anything reads it.
PLATFORM_PATTERN = r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)?$"


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


def compose_system_tag(workspace: str, image: str) -> str:
    """``<workspace>--<image>``: the tag a build pushes for one image.

    Caller tags can't contain ``--``, so no caller can push one.
    """
    if "--" in workspace or "--" in image:
        raise ImageIdentityError(f"workspace and image must not contain '--', got {workspace!r} and {image!r}")
    tag = f"{workspace}--{image}"
    if not re.fullmatch(TAG_PATTERN, tag):
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
    """``value`` as BuildKit names that platform, for a model's field validator.

    Raises:
        ImageIdentityError: ``value`` names no platform.
    """
    return str(Platform.parse(value)) if isinstance(value, str) else value
