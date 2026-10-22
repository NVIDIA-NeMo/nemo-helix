# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a service's image comes from: a published reference, or a build from the archive."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_builder_plugin.identity import ImageIdentityError, normalize_reference
from nemo_evals.harbor.compose.models import ComposeBuild, ComposeError, ComposeImage, ComposeProblems
from nemo_evals.harbor.compose.resolve import ENVIRONMENT_DIR, path_in_environment
from nemo_evals.harbor.compose.service_options import parse_environment

_BUILD_KEYS = frozenset({"context", "dockerfile", "args", "target", "no_cache"})

# A build context that names a URL or a git remote instead of a directory.
_REMOTE_CONTEXT = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://|^git@")


def normalize_image(reference: str) -> str:
    """Normalize an image reference as Docker does: ``postgres:18`` is ``docker.io/library/postgres:18``.

    A reference with no tag gets ``:latest``. A digest is kept, and so is a tag written next to it, as
    ``docker pull`` accepts both.

    Raises:
        ComposeError: ``reference`` isn't a valid image reference.
    """
    try:
        parsed = normalize_reference(reference)
    except ImageIdentityError as error:
        raise ComposeError("compose_invalid", f"image {reference!r}: {error}") from error

    # Docker accepts `name:tag@digest`; keep the tag beside the digest when the reference has both.
    name, digest_separator, _ = reference.strip().partition("@")
    _, tag_separator, tag = name.rpartition(":")
    if digest_separator and tag_separator and "/" not in tag:
        return f"{parsed.registry}/{parsed.repository}:{tag}@{parsed.digest}"
    return str(parsed)


def parse_image_source(
    name: str, raw_service: Mapping[str, Any], problems: ComposeProblems
) -> ComposeImage | ComposeBuild | None:
    """Read where service ``name``'s image comes from; ``build`` wins when both ``build`` and ``image`` are set, as in Compose.

    Returns None, with the problem recorded, when the service has neither or its reference is invalid.
    """
    path = f"services.{name}"
    build = raw_service.get("build")
    if build is not None:
        return parse_build_source(name, build, problems)

    image = raw_service.get("image")
    if not isinstance(image, str) or not image:
        problems.invalid(path, "has neither `image` nor `build`")
        return None

    try:
        return ComposeImage(ref=normalize_image(image))
    except ComposeError as error:
        problems.invalid(f"{path}.image", error.detail)
        return None


def parse_build_source(name: str, build: Any, problems: ComposeProblems) -> ComposeBuild | None:
    """Read service ``name``'s ``build`` section as a context under ``environment/`` and a Dockerfile inside it.

    ``{context: ./api, dockerfile: Dockerfile.prod}`` gives the context ``environment/api``. A remote context, a
    context outside ``environment/`` and a Dockerfile outside its context are unsupported.
    """
    path = f"services.{name}.build"
    if not isinstance(build, dict):
        problems.invalid(path, "expected a mapping or a path")
        return None

    for key in build:
        if key not in _BUILD_KEYS:
            problems.unsupported(f"{path}.{key}")

    context = build.get("context", ".")
    dockerfile = build.get("dockerfile", "Dockerfile")
    if not isinstance(context, str) or not isinstance(dockerfile, str):
        problems.invalid(path, "`context` and `dockerfile` must be strings")
        return None
    if _REMOTE_CONTEXT.match(context):
        problems.unsupported(f"{path}.context: remote context")
        return None

    # Contexts resolve against `environment/`; neither they nor the Dockerfile may reach outside the task.
    context_path = path_in_environment(context)
    if context_path is None:
        problems.unsupported(f"{path}.context: {context} is outside environment/")
        return None
    if posixpath.isabs(dockerfile) or ".." in dockerfile.split("/"):
        problems.unsupported(f"{path}.dockerfile: {dockerfile} is outside its context")
        return None

    target = build.get("target")
    return ComposeBuild(
        context_path=context_path,
        dockerfile=dockerfile,
        args=parse_environment(build.get("args"), f"{path}.args", problems),
        target=target if isinstance(target, str) else None,
        no_cache=build.get("no_cache") is True,
    )


def dockerfile_path(build: ComposeBuild | None) -> str | None:
    """Return a build's Dockerfile relative to ``environment/`` (``api/Dockerfile``); None without a build."""
    if build is None:
        return None
    return posixpath.relpath(posixpath.join(build.context_path, build.dockerfile), ENVIRONMENT_DIR)


def dockerfile_entrypoint(text: str) -> list[str] | None:
    """Find the ``ENTRYPOINT`` the last build stage of a Dockerfile ends with: exec form as written, shell form as ``sh -c``.

    A new ``FROM`` starts over, since its base image's entrypoint isn't known here. Given
    ``ENTRYPOINT ["/start.sh"]`` this returns ``["/start.sh"]``; given ``ENTRYPOINT /start.sh`` it returns
    ``["sh", "-c", "/start.sh"]``.
    """
    entrypoint: list[str] | None = None
    continued = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Join a continued instruction (trailing backslash) into one line before reading it.
        if stripped.endswith("\\"):
            continued += stripped[:-1] + " "
            continue

        instruction = (continued + stripped).strip()
        continued = ""

        keyword, _, arguments = instruction.partition(" ")
        if keyword.upper() == "FROM":
            entrypoint = None
        elif keyword.upper() == "ENTRYPOINT":
            entrypoint = _parse_entrypoint_arguments(arguments.strip())
    return entrypoint


def read_dockerfile_entrypoint(
    environment_dir: Path, dockerfile: str, path: str, problems: ComposeProblems
) -> list[str] | None:
    """Read the ``ENTRYPOINT`` of the Dockerfile at ``dockerfile`` (relative to ``environment/``).

    A Dockerfile that isn't there or can't be read is recorded against ``path``.
    """
    file = environment_dir / dockerfile
    if not file.is_file():
        problems.invalid(path, f"main's Dockerfile {dockerfile} is not in the task")
        return None

    try:
        return dockerfile_entrypoint(file.read_text(encoding="utf-8", errors="replace"))
    except OSError as error:
        problems.invalid(path, f"can't read main's Dockerfile: {error}")
        return None


def _parse_entrypoint_arguments(arguments: str) -> list[str] | None:
    """Parse what follows ``ENTRYPOINT``: a JSON array is exec form, anything else is shell form (run with ``sh -c``)."""
    if arguments.startswith("["):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed or None
    return ["sh", "-c", arguments] if arguments else None
