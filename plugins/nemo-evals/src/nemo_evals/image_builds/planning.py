# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which container images a Harbor task needs, from its ``task.toml`` and archive."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from harbor.models.task.config import EnvironmentConfig, TaskConfig, TaskOS
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config, task_has_any_separate_verifier
from nemo_builder_plugin.identity import ImageIdentityError, Platform, normalize_reference, normalize_registry
from nemo_evals.harbor.compose.models import ComposeBuild, ComposeService, ComposeTask
from nemo_evals.image_builds.models import BuildImage
from pydantic import ValidationError

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")
_MAX_DOCKERFILE_BYTES = 1024 * 1024
# `FROM [--platform=<p>] <image> [AS <name>]`, the flag spelled any way BuildKit accepts it.
_FROM = re.compile(r"^\s*FROM\s+(?P<flags>(?:--\S+\s+)*)", re.IGNORECASE)
_PLATFORM_FLAG = re.compile(r"--platform=(\S+)", re.IGNORECASE)


class ImagePlanError(ValueError):
    """A task whose images can't be built. ``code`` is stable and shown to users."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class TaskArchive:
    """The file listing of a published task archive, which holds one task directory."""

    root: str
    files: frozenset[str]
    compose: str | None
    #: The archive itself, to read a Dockerfile from; None when there is nothing to read.
    path: Path | None = None
    compose_task: ComposeTask | None = field(default=None, compare=False, hash=False)
    """The compose file as ``parse_compose`` reads it. The runner sets it whenever ``compose`` is."""

    @classmethod
    def read(cls, path: Path) -> TaskArchive:
        with tarfile.open(path, "r:*") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            roots = {member.name.split("/", 1)[0] for member in members}
            if len(roots) != 1:
                raise ImagePlanError("invalid_task_archive", "the archive must hold exactly one task directory")
            (root,) = roots
            compose = None
            compose_paths = [f"{root}/environment/{name}" for name in COMPOSE_FILENAMES]
            compose_member = next((m for path in compose_paths for m in members if m.name == path), None)
            if compose_member is not None:
                handle = archive.extractfile(compose_member)
                if handle is not None:
                    compose = handle.read().decode()
        return cls(root=root, files=frozenset(member.name for member in members), compose=compose, path=path)

    def has(self, relative_path: str) -> bool:
        return f"{self.root}/{relative_path}" in self.files

    def context_digest(self, context_path: str) -> str | None:
        """A digest of everything under ``context_path``: each entry's path, kind, mode, link target and contents.

        Two tasks whose build contexts hold the same files get the same digest, whatever else their archives hold,
        so they can share one build. None when the archive can't be read here.
        """
        if self.path is None:
            return None
        prefix = f"{self.root}/{context_path}/"
        entries = []
        with tarfile.open(self.path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.name.startswith(prefix):
                    continue
                content = ""
                if member.isfile():
                    handle = archive.extractfile(member)
                    content = hashlib.sha256(handle.read() if handle is not None else b"").hexdigest()
                entries.append(
                    [
                        member.name.removeprefix(prefix),
                        member.type.decode(),
                        member.mode & 0o7777,
                        member.linkname,
                        content,
                    ]
                )
        entries.sort()
        return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()

    def text(self, relative_path: str) -> str | None:
        """A file of the task as text, or None when the archive can't be read here or the file isn't text."""
        if self.path is None:
            return None
        with tarfile.open(self.path, "r:*") as archive:
            try:
                handle = archive.extractfile(f"{self.root}/{relative_path}")
            except KeyError:
                return None
            if handle is None:
                return None
            try:
                return handle.read(_MAX_DOCKERFILE_BYTES).decode()
            except UnicodeDecodeError:
                return None


def parse_task_config(config: Mapping[str, Any]) -> TaskConfig:
    """Validate a stored ``task.toml`` and refuse what can't be built at all, before any build starts."""
    try:
        task_config = TaskConfig.model_validate(config)
    except ValidationError as exc:
        raise ImagePlanError("invalid_task_config", str(exc)) from exc
    environments = [task_config.environment]
    if task_config.verifier.environment is not None:
        environments.append(task_config.verifier.environment)
    if any(environment.os == TaskOS.WINDOWS for environment in environments):
        raise ImagePlanError("windows_unsupported", "Windows task environments can't be built")
    if task_config.steps and task_has_any_separate_verifier(task_config):
        raise ImagePlanError(
            "multi_step_separate_verifier_unsupported", "multi-step tasks with a separate verifier can't be built"
        )
    return task_config


#: Harbor's name for the agent's container, the one service every Docker task has.
MAIN_SERVICE = "main"
VERIFIER_TYPE = "verifier"
#: Prefixes a compose service's name in its image's type. Compose service names can't contain `/`, so no service's
#: type can collide with `main` or `verifier`, whatever the service is called.
SERVICE_TYPE_PREFIX = "service/"


def service_image_type(name: str) -> str:
    """The type of compose service ``name``'s image: ``db`` gives ``service/db``."""
    return f"{SERVICE_TYPE_PREFIX}{name}"


def plan_images(config: Mapping[str, Any], archive: TaskArchive, *, allowed_registries: list[str]) -> list[BuildImage]:
    """Every image the task needs, in the order ``main``, its compose services, then a separate verifier.

    Each image's type says which container runs it: ``main``, ``verifier`` for a separate verifier, and
    ``service/<name>`` for each compose service, so a service may have any name, ``verifier`` included.
    """
    task_config = parse_task_config(config)
    compose = _compose_task(archive)
    main_platform = compose.main.platform if compose is not None else None
    images = [_pinned(_main_image(task_config.environment, archive, compose), MAIN_SERVICE, main_platform)]
    if compose is not None:
        images += [
            _pinned(_service_image(name, service, archive), name, service.platform)
            for name, service in compose.services.items()
        ]
    verifier = resolve_effective_verifier_env_config(task_config, None)
    if verifier is not None:
        images.append(_verifier_image(verifier, archive))
    allowed = {normalize_registry(host) for host in allowed_registries}
    for image in images:
        if image.image is None:
            continue
        try:
            registry = normalize_reference(image.image).registry
        except ImageIdentityError as exc:
            raise ImagePlanError("invalid_image_reference", f"{image.type} image {image.image!r}: {exc}") from exc
        if allowed and registry not in allowed:
            raise ImagePlanError(
                "registry_not_allowed",
                f"{image.type} image {image.image} is from {registry}, which is not in allowed_registries",
            )
    return images


def _compose_task(archive: TaskArchive) -> ComposeTask | None:
    """The task's compose file as the parser reads it, so the services planned are the services that will run."""
    if archive.compose is not None and archive.compose_task is None:
        raise ImagePlanError("invalid_task_archive", "the task's compose file wasn't read")
    return archive.compose_task


def _main_image(environment: EnvironmentConfig, archive: TaskArchive, compose: ComposeTask | None) -> BuildImage:
    """``main`` as Harbor's base compose file makes it, with the task's compose file laid over it.

    The base's ``main`` is the task's ``docker_image``, or a build of ``environment/`` with ``Dockerfile``. Over a
    ``docker_image``, the task's ``main.image`` replaces it and its ``main.build`` is never built: a trial starts
    without ``force_build``, so Harbor runs the prebuilt image. Over a build, the task's ``main.build`` overrides the
    fields it sets, and its ``main.image`` only names the image built. The parser has already applied all of that.
    """
    if environment.docker_image:
        replacement = compose.main_image if compose is not None else None
        return BuildImage(type=MAIN_SERVICE, source="prebuilt", image=str(replacement or environment.docker_image))

    build = compose.main_build if compose is not None else None
    if build is None:
        context, dockerfile = "environment", "Dockerfile"
    else:
        _check_build_options(MAIN_SERVICE, build)
        context, dockerfile = build.context_path, build.dockerfile

    is_default_build = (context, dockerfile) == ("environment", "Dockerfile")
    if is_default_build and not archive.has("environment/Dockerfile"):
        raise ImagePlanError("main_dockerfile_missing", "the task names no image and has no environment/Dockerfile")
    return _dockerfile_image(MAIN_SERVICE, archive, context, dockerfile)


def _service_image(name: str, service: ComposeService, archive: TaskArchive) -> BuildImage:
    """A compose service: built from the archive, or a published image the build copies."""
    source = service.source
    if isinstance(source, ComposeBuild):
        _check_build_options(name, source)
        return _dockerfile_image(service_image_type(name), archive, source.context_path, source.dockerfile)
    return BuildImage(type=service_image_type(name), source="prebuilt", image=source.reference)


def _check_build_options(service: str, build: ComposeBuild) -> None:
    """Refuse a build that sets options the image builder can't apply.

    Builder builds a context and a Dockerfile, nothing else: building without ``args``, ``target`` or ``no_cache``
    would make a different image than the task asked for, and report it ready.
    """
    options = {"args": bool(build.args), "no_cache": build.no_cache, "target": build.target is not None}
    if unsupported := [option for option, is_set in options.items() if is_set]:
        raise ImagePlanError(
            "unsupported_compose_build_option",
            f"service {service} sets build options the image builder can't apply: {', '.join(unsupported)}",
        )


def _verifier_image(environment: EnvironmentConfig, archive: TaskArchive) -> BuildImage:
    if environment.docker_image:
        return BuildImage(type=VERIFIER_TYPE, source="prebuilt", image=environment.docker_image)
    if archive.has("tests/Dockerfile"):
        return _dockerfile_image(VERIFIER_TYPE, archive, "tests", "Dockerfile")
    raise ImagePlanError(
        "verifier_dockerfile_missing", "the separate verifier names no image and the task has no tests/Dockerfile"
    )


def _pinned(image: BuildImage, service: str, platform: str | None) -> BuildImage:
    """``image`` for the platform its compose service names, which compose builds or pulls for over any other pin."""
    pin = _compose_pin(service, platform)
    return image if pin is None else image.model_copy(update={"platform": pin})


def _compose_pin(service: str, platform: str | None) -> str | None:
    """The platform a compose service names; the parser has already filled in any variable it used."""
    if platform is None:
        return None
    try:
        return str(Platform.parse(platform))
    except ImageIdentityError:
        return _unreadable(service, platform)


def _unreadable(service: str, platform: object) -> NoReturn:
    raise ImagePlanError("invalid_compose_file", f"service {service} names an unreadable platform: {platform!r}")


def dockerfile_pin(dockerfile: str) -> str | None:
    """The platform a Dockerfile's final stage names plainly, as in ``FROM --platform=linux/amd64 ...``.

    The final stage is the image. A platform given through a variable, such as ``$BUILDPLATFORM``, pins nothing.
    """
    # A comment line never continues an instruction, even ending in a backslash, and drops out of one it splits.
    instructions = "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#"))
    lines = re.sub(r"\\\n", " ", instructions).splitlines()
    froms = [match for line in lines if (match := _FROM.match(line))]
    if not froms:
        return None
    flag = _PLATFORM_FLAG.search(froms[-1].group("flags"))
    if flag is None:
        return None
    try:
        return str(Platform.parse(flag.group(1)))
    except ImageIdentityError:
        # A variable, such as `$BUILDPLATFORM`, isn't a platform.
        return None


def _dockerfile_image(image_type: str, archive: TaskArchive, context_path: str, dockerfile: str) -> BuildImage:
    path = posixpath.normpath(posixpath.join(context_path, dockerfile))
    if not archive.has(path):
        raise ImagePlanError("dockerfile_missing", f"{image_type} has no {context_path}/{dockerfile}")
    text = archive.text(path)
    return BuildImage(
        type=image_type,
        source="dockerfile",
        context_path=f"{archive.root}/{context_path}",
        dockerfile=dockerfile,
        context_digest=archive.context_digest(context_path),
        platform=dockerfile_pin(text) if text is not None else None,
    )
