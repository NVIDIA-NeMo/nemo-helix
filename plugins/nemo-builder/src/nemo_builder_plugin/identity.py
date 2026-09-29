# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, repository paths and tags, and the system tag."""

from __future__ import annotations

import re

#: A sha256 digest, in lowercase hex only, so that one value has one spelling.
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"

#: One component of a repository path.
REPOSITORY_COMPONENT_PATTERN = r"^[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*$"

_REGISTRY = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?")
_REPOSITORY_COMPONENT = re.compile(REPOSITORY_COMPONENT_PATTERN)
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
    """A tag a caller may push: valid, and not shaped like a system tag or a signature tag.

    A caller tag equal to another image's system tag would be what the broker signs for that image.
    One shaped like cosign's ``sha256-<hex>.sig`` would replace an image's signature.
    """
    validate_tag(tag)
    if "--" in tag:
        raise ImageIdentityError(f"tag {tag!r} contains '--', which is reserved for system tags")
    if tag.startswith("sha256-"):
        raise ImageIdentityError(f"tag {tag!r} starts with 'sha256-', which is reserved for signatures")
    return tag


def validate_registry_host(registry: str) -> str:
    """A registry host, optionally with a port, lowercased."""
    host = registry.strip().lower()
    if not _REGISTRY.fullmatch(host):
        raise ImageIdentityError(f"invalid registry host: {registry!r}")
    return host


def compose_system_tag(workspace: str, build_set: str, revision: int, index: int) -> str:
    """``<workspace>--<build_set>-<revision>-<index>``: the tag a build pushes and the broker signs.

    One per image, so that two specs publishing to one repository do not replace each other's. A
    workspace or set name cannot contain ``--``, and a caller tag cannot either, so the tag splits
    unambiguously and no caller can push one.
    """
    if revision < 1:
        raise ImageIdentityError("revision must be >= 1")
    if index < 0:
        raise ImageIdentityError("index must be >= 0")
    if "--" in workspace or "--" in build_set:
        raise ImageIdentityError(f"workspace and build_set must not contain '--', got {workspace!r} and {build_set!r}")
    tag = f"{workspace}--{build_set}-{revision}-{index}"
    if not _TAG.fullmatch(tag):
        raise ImageIdentityError(f"composed system tag is not a valid tag: {tag!r}")
    return tag


def split_system_tag(tag: str) -> tuple[str, str, int, int]:
    """Inverse of :func:`compose_system_tag`."""
    if tag.count("--") != 1:
        raise ImageIdentityError(f"not a system tag: {tag!r}")
    workspace, rest = tag.split("--", 1)
    rest, _, index = rest.rpartition("-")
    set_name, _, revision = rest.rpartition("-")
    if not set_name or not revision.isdigit() or not index.isdigit():
        raise ImageIdentityError(f"not a system tag: {tag!r}")
    return workspace, set_name, int(revision), int(index)
