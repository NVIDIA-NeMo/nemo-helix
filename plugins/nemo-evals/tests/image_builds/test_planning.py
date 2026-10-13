# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which images a Harbor task needs, decided from ``task.toml`` and the archive listing."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from nemo_evals.image_builds.planning import ImagePlanError, TaskArchive, plan_images, registry_host


def _archive(*paths: str, compose: str | None = None) -> TaskArchive:
    files = {f"sort-task/{path}" for path in paths}
    if compose is not None:
        files.add("sort-task/environment/docker-compose.yaml")
    return TaskArchive(root="sort-task", files=frozenset(files), compose=compose)


def _plan(config: dict, archive: TaskArchive, allowed: list[str] | None = None) -> list[tuple]:
    return [
        (image.role, image.source, image.context_path, image.dockerfile, image.image)
        for image in plan_images(config, archive, allowed_registries=allowed or [])
    ]


def test_dockerfile_agent_with_shared_verifier_needs_one_image() -> None:
    assert _plan({}, _archive("environment/Dockerfile")) == [
        ("agent", "dockerfile", "sort-task/environment", "Dockerfile", None)
    ]


def test_separate_verifier_builds_from_tests() -> None:
    config = {"verifier": {"environment_mode": "separate"}}

    assert _plan(config, _archive("environment/Dockerfile", "tests/Dockerfile")) == [
        ("agent", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("verifier", "dockerfile", "sort-task/tests", "Dockerfile", None),
    ]


def test_separate_verifier_without_its_own_environment_reuses_a_prebuilt_agent_image() -> None:
    """Harbor 0.20 copies ``[environment]`` for the verifier, so a named agent image wins over tests/."""
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment_mode": "separate"}}

    assert _plan(config, _archive("tests/Dockerfile"), ["ghcr.io"]) == [
        ("agent", "prebuilt", None, None, "ghcr.io/org/sort:1"),
        ("verifier", "prebuilt", None, None, "ghcr.io/org/sort:1"),
    ]


def test_explicit_verifier_environment_without_an_image_builds_tests() -> None:
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment": {"cpus": 1}}}

    assert _plan(config, _archive("tests/Dockerfile"), ["ghcr.io"])[1] == (
        "verifier",
        "dockerfile",
        "sort-task/tests",
        "Dockerfile",
        None,
    )


def test_compose_services_are_each_an_image() -> None:
    compose = """
services:
  main:
    build: .
  db:
    build:
      context: ./db
      dockerfile: Dockerfile.db
  cache:
    image: redis:7
"""
    archive = _archive("environment/Dockerfile", "environment/db/Dockerfile.db", compose=compose)

    assert _plan({}, archive, ["docker.io"]) == [
        ("service:main", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("service:db", "dockerfile", "sort-task/environment/db", "Dockerfile.db", None),
        ("service:cache", "prebuilt", None, None, "redis:7"),
    ]


@pytest.mark.parametrize(
    ("config", "archive", "code"),
    [
        ({"environment": {"os": "windows"}}, _archive("environment/Dockerfile"), "windows_unsupported"),
        ({}, _archive("instruction.md"), "agent_dockerfile_missing"),
        (
            {"verifier": {"environment_mode": "separate"}},
            _archive("environment/Dockerfile"),
            "verifier_dockerfile_missing",
        ),
        ({"environment": {"docker_image": "quay.io/org/x:1"}}, _archive(), "registry_not_allowed"),
        ({}, _archive(compose="services:\n  main:\n    build: ../../elsewhere\n"), "invalid_compose_file"),
        ({}, _archive(compose="services: {}\n"), "invalid_compose_file"),
        ({}, _archive(compose="services:\n  main:\n    build: ./missing\n"), "dockerfile_missing"),
        ({"environment": {"cpus": "lots"}}, _archive("environment/Dockerfile"), "invalid_task_config"),
    ],
)
def test_unbuildable_tasks_fail_with_a_stable_code(config: dict, archive: TaskArchive, code: str) -> None:
    with pytest.raises(ImagePlanError) as error:
        plan_images(config, archive, allowed_registries=["docker.io"])

    assert error.value.code == code


@pytest.mark.parametrize(
    ("image", "host"),
    [
        ("ubuntu:24.04", "docker.io"),
        ("library/ubuntu", "docker.io"),
        ("ghcr.io/org/app:1", "ghcr.io"),
        ("localhost/app", "localhost"),
        ("registry:5000/app", "registry:5000"),
    ],
)
def test_registry_host(image: str, host: str) -> None:
    assert registry_host(image) == host


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_task_archive_reads_listing_and_compose(tmp_path: Path) -> None:
    path = tmp_path / "task_archive"
    _write_tar(
        path,
        {
            "sort-task/task.toml": b"",
            "sort-task/environment/Dockerfile": b"FROM python:3.12\n",
            "sort-task/environment/docker-compose.yaml": b"services: {}\n",
        },
    )

    archive = TaskArchive.read(path)

    assert archive.root == "sort-task"
    assert archive.has("environment/Dockerfile")
    assert archive.compose == "services: {}\n"


def test_task_archive_reads_a_yml_compose_file(tmp_path: Path) -> None:
    path = tmp_path / "task_archive"
    _write_tar(path, {"sort-task/environment/docker-compose.yml": b"services: {}\n"})

    assert TaskArchive.read(path).compose == "services: {}\n"


def test_task_archive_with_two_task_directories_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "task_archive"
    _write_tar(path, {"a/task.toml": b"", "b/task.toml": b""})

    with pytest.raises(ImagePlanError) as error:
        TaskArchive.read(path)

    assert error.value.code == "invalid_task_archive"
