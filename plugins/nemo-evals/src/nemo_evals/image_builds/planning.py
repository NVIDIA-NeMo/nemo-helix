# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which container images a Harbor task needs, from its ``task.toml`` and archive."""

from __future__ import annotations

import posixpath
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.models.task.config import EnvironmentConfig, TaskConfig, TaskOS
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config, task_has_any_separate_verifier
from nemo_builder_plugin.identity import ImageIdentityError, normalize_reference, normalize_registry
from nemo_evals.image_builds.models import BuildImage
from pydantic import ValidationError

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")


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
        return cls(root=root, files=frozenset(member.name for member in members), compose=compose)

    def has(self, relative_path: str) -> bool:
        return f"{self.root}/{relative_path}" in self.files


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
VERIFIER_ROLE = "verifier"


def plan_images(config: Mapping[str, Any], archive: TaskArchive, *, allowed_registries: list[str]) -> list[BuildImage]:
    """Every image the task needs, in the order ``main``, its compose services, then a separate verifier.

    Images are named as Harbor names the containers: ``main``, each compose service by its own name, and
    ``verifier``.
    """
    task_config = parse_task_config(config)
    services = _compose_services(archive)
    images = [_main_image(task_config.environment, archive, services.get(MAIN_SERVICE) or {})]
    images += [_service_image(name, service, archive) for name, service in services.items() if name != MAIN_SERVICE]
    verifier = resolve_effective_verifier_env_config(task_config, None)
    if verifier is not None:
        if VERIFIER_ROLE in services:
            raise ImagePlanError(
                "invalid_compose_file", "a compose service named verifier can't run next to a separate verifier"
            )
        images.append(_verifier_image(verifier, archive))
    allowed = {normalize_registry(host) for host in allowed_registries}
    for image in images:
        if image.image is None:
            continue
        try:
            registry = normalize_reference(image.image).registry
        except ImageIdentityError as exc:
            raise ImagePlanError("invalid_image_reference", f"{image.role} image {image.image!r}: {exc}") from exc
        if allowed and registry not in allowed:
            raise ImagePlanError(
                "registry_not_allowed",
                f"{image.role} image {image.image} is from {registry}, which is not in allowed_registries",
            )
    return images


def _compose_services(archive: TaskArchive) -> dict[str, dict]:
    """The task's compose services, by name. Harbor lays this file over its own, so it may leave out ``main``."""
    if archive.compose is None:
        return {}
    try:
        compose = yaml.safe_load(archive.compose)
    except yaml.YAMLError as exc:
        raise ImagePlanError("invalid_compose_file", str(exc)) from exc
    if compose is None:
        return {}
    services = compose.get("services", {}) if isinstance(compose, dict) else None
    if not isinstance(services, dict):
        raise ImagePlanError("invalid_compose_file", "the environment's compose file has unreadable services")
    for name, service in services.items():
        if service is not None and not isinstance(service, dict):
            raise ImagePlanError("invalid_compose_file", f"service {name} isn't a mapping")
    return {str(name): service or {} for name, service in services.items()}


def _main_image(environment: EnvironmentConfig, archive: TaskArchive, overlay: dict) -> BuildImage:
    """``main`` as Harbor's base compose file makes it, with the task's compose file laid over it.

    The base's ``main`` is the task's ``docker_image``, or a build of ``environment/`` with ``Dockerfile``. Over a
    ``docker_image``, the task's ``main.image`` replaces it and its ``main.build`` is never built: a trial starts
    without ``force_build``, so Harbor runs the prebuilt image. Over a build, the task's ``main.build`` overrides the
    fields it sets, and its ``main.image`` only names the image built.
    """
    if environment.docker_image:
        image = overlay.get("image") or environment.docker_image
        return BuildImage(role=MAIN_SERVICE, source="prebuilt", image=str(image))
    build = overlay.get("build")
    if build is None:
        if not archive.has("environment/Dockerfile"):
            raise ImagePlanError("main_dockerfile_missing", "the task names no image and has no environment/Dockerfile")
        return _dockerfile_image(MAIN_SERVICE, archive, "environment", "Dockerfile")
    return _compose_build(MAIN_SERVICE, build, archive)


def _service_image(name: str, service: dict, archive: TaskArchive) -> BuildImage:
    build = service.get("build")
    if build is not None:
        return _compose_build(name, build, archive)
    image = service.get("image")
    if not isinstance(image, str) or not image:
        raise ImagePlanError("invalid_compose_file", f"service {name} has neither build nor image")
    return BuildImage(role=name, source="prebuilt", image=image)


def _compose_build(name: str, build: object, archive: TaskArchive) -> BuildImage:
    """A service built as its compose ``build`` says; relative paths are from ``environment/``, as in compose."""
    context, dockerfile = (build, "Dockerfile") if isinstance(build, str) else _build_options(name, build)
    if posixpath.isabs(dockerfile) or ".." in dockerfile.split("/"):
        raise ImagePlanError("invalid_compose_file", f"service {name} names a dockerfile outside its context")
    context_path = posixpath.normpath(posixpath.join("environment", context))
    if context_path == ".." or context_path.startswith("../") or posixpath.isabs(context_path):
        raise ImagePlanError("invalid_compose_file", f"service {name} builds from outside the task: {context}")
    return _dockerfile_image(name, archive, context_path, dockerfile)


def _verifier_image(environment: EnvironmentConfig, archive: TaskArchive) -> BuildImage:
    if environment.docker_image:
        return BuildImage(role=VERIFIER_ROLE, source="prebuilt", image=environment.docker_image)
    if archive.has("tests/Dockerfile"):
        return _dockerfile_image(VERIFIER_ROLE, archive, "tests", "Dockerfile")
    raise ImagePlanError(
        "verifier_dockerfile_missing", "the separate verifier names no image and the task has no tests/Dockerfile"
    )


def _build_options(service: str, build: object) -> tuple[str, str]:
    if not isinstance(build, dict) or any(not isinstance(key, str) for key in build):
        raise ImagePlanError("invalid_compose_file", f"service {service} has an unreadable build section")
    context = build.get("context", ".")
    dockerfile = build.get("dockerfile", "Dockerfile")
    if not isinstance(context, str) or not isinstance(dockerfile, str):
        raise ImagePlanError("invalid_compose_file", f"service {service} has an unreadable build section")
    unsupported = sorted(set(build) - {"context", "dockerfile"})
    if unsupported:
        raise ImagePlanError(
            "unsupported_compose_build_option",
            f"service {service} sets build options the image builder can't apply: {', '.join(unsupported)}",
        )
    return context, dockerfile


def _dockerfile_image(role: str, archive: TaskArchive, context_path: str, dockerfile: str) -> BuildImage:
    if not archive.has(posixpath.normpath(posixpath.join(context_path, dockerfile))):
        raise ImagePlanError("dockerfile_missing", f"{role} has no {context_path}/{dockerfile}")
    return BuildImage(
        role=role, source="dockerfile", context_path=f"{archive.root}/{context_path}", dockerfile=dockerfile
    )
