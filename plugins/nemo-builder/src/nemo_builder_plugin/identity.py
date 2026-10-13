# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, repository paths and tags, and the system tag."""

from __future__ import annotations

import re

#: Lowercase hex only, so that one digest has one spelling.
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"

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
