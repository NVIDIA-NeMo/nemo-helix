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
from nemo_evals.image_builds.models import BuildImage
from pydantic import ValidationError

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")
_DEFAULT_REGISTRY = "docker.io"


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


def plan_images(config: Mapping[str, Any], archive: TaskArchive, *, allowed_registries: list[str]) -> list[BuildImage]:
    """Every image the task needs, in the order agent (or compose services), then verifier."""
    task_config = parse_task_config(config)
    images = _agent_images(task_config.environment, archive)
    verifier = resolve_effective_verifier_env_config(task_config, None)
    if verifier is not None:
        images.append(_verifier_image(verifier, archive))
    for image in images:
        if image.image is not None and registry_host(image.image) not in allowed_registries:
            raise ImagePlanError(
                "registry_not_allowed",
                f"{image.role} image {image.image} is from {registry_host(image.image)}, "
                f"which is not in allowed_registries",
            )
    return images


def _agent_images(environment: EnvironmentConfig, archive: TaskArchive) -> list[BuildImage]:
    if environment.docker_image:
        return [BuildImage(role="agent", source="prebuilt", image=environment.docker_image)]
    if archive.compose is not None:
        return _compose_images(archive)
    if archive.has("environment/Dockerfile"):
        return [_dockerfile_image("agent", archive, "environment", "Dockerfile")]
    raise ImagePlanError("agent_dockerfile_missing", "the task names no image and has no environment/Dockerfile")


def _verifier_image(environment: EnvironmentConfig, archive: TaskArchive) -> BuildImage:
    if environment.docker_image:
        return BuildImage(role="verifier", source="prebuilt", image=environment.docker_image)
    if archive.has("tests/Dockerfile"):
        return _dockerfile_image("verifier", archive, "tests", "Dockerfile")
    raise ImagePlanError(
        "verifier_dockerfile_missing", "the separate verifier names no image and the task has no tests/Dockerfile"
    )


def _compose_images(archive: TaskArchive) -> list[BuildImage]:
    try:
        compose = yaml.safe_load(archive.compose or "")
    except yaml.YAMLError as exc:
        raise ImagePlanError("invalid_compose_file", str(exc)) from exc
    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict) or not services:
        raise ImagePlanError("invalid_compose_file", "the environment's compose file has no services")
    images: list[BuildImage] = []
    for name, service in services.items():
        role = f"service:{name}"
        service = service if isinstance(service, dict) else {}
        build = service.get("build")
        if build is None:
            image = service.get("image")
            if not isinstance(image, str) or not image:
                raise ImagePlanError("invalid_compose_file", f"service {name} has neither build nor image")
            images.append(BuildImage(role=role, source="prebuilt", image=image))
            continue
        context, dockerfile = (build, "Dockerfile") if isinstance(build, str) else _build_options(name, build)
        if posixpath.isabs(dockerfile) or ".." in dockerfile.split("/"):
            raise ImagePlanError("invalid_compose_file", f"service {name} names a dockerfile outside its context")
        context_path = posixpath.normpath(posixpath.join("environment", context))
        if context_path == ".." or context_path.startswith("../") or posixpath.isabs(context_path):
            raise ImagePlanError("invalid_compose_file", f"service {name} builds from outside the task: {context}")
        images.append(_dockerfile_image(role, archive, context_path, dockerfile))
    return images


def _build_options(service: str, build: object) -> tuple[str, str]:
    if not isinstance(build, dict):
        raise ImagePlanError("invalid_compose_file", f"service {service} has an unreadable build section")
    context = build.get("context", ".")
    dockerfile = build.get("dockerfile", "Dockerfile")
    if not isinstance(context, str) or not isinstance(dockerfile, str):
        raise ImagePlanError("invalid_compose_file", f"service {service} has an unreadable build section")
    return context, dockerfile


def _dockerfile_image(role: str, archive: TaskArchive, context_path: str, dockerfile: str) -> BuildImage:
    if not archive.has(posixpath.normpath(posixpath.join(context_path, dockerfile))):
        raise ImagePlanError("dockerfile_missing", f"{role} has no {context_path}/{dockerfile}")
    return BuildImage(
        role=role, source="dockerfile", context_path=f"{archive.root}/{context_path}", dockerfile=dockerfile
    )


def registry_host(image: str) -> str:
    """The registry an image reference pulls from, with Docker Hub for a bare name."""
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        return first
    return _DEFAULT_REGISTRY
