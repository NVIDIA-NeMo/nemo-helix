# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which images a Harbor task needs, decided from ``task.toml`` and the archive listing."""

from __future__ import annotations

import io
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from nemo_evals.harbor.compose.models import ComposeError
from nemo_evals.harbor.compose.parser import parse_compose
from nemo_evals.image_builds.planning import ImagePlanError, TaskArchive, dockerfile_pin, plan_images


def _archive(*paths: str, compose: str | None = None) -> TaskArchive:
    files = {f"sort-task/{path}" for path in paths}
    if compose is not None:
        files.add("sort-task/environment/docker-compose.yaml")
    return TaskArchive(root="sort-task", files=frozenset(files), compose=compose)


def _parsed(config: dict, archive: TaskArchive) -> TaskArchive:
    """The archive with its compose file parsed, as ``read_task_archive`` does."""
    if archive.compose is None:
        return archive

    with tempfile.TemporaryDirectory() as scratch:
        task = Path(scratch) / archive.root
        (task / "environment").mkdir(parents=True)
        (task / "environment" / "docker-compose.yaml").write_text(archive.compose, encoding="utf-8")
        image = (config.get("environment") or {}).get("docker_image")
        if image:
            (task / "task.toml").write_text(f'[environment]\ndocker_image = "{image}"\n', encoding="utf-8")

        try:
            return replace(archive, compose_task=parse_compose(task, read_entrypoint=False))
        except ComposeError as exc:
            raise ImagePlanError("invalid_compose_file", str(exc)) from exc


def _plan_images(config: dict, archive: TaskArchive, allowed_registries: list[str]):
    """Plan a task's images with its compose file parsed first, as ``read_task_archive`` would have."""
    return plan_images(config, _parsed(config, archive), allowed_registries=allowed_registries)


def _plan(config: dict, archive: TaskArchive, allowed: list[str] | None = None) -> list[tuple]:
    return [
        (image.type, image.source, image.context_path, image.dockerfile, image.image)
        for image in _plan_images(config, archive, allowed or [])
    ]


def test_dockerfile_main_with_shared_verifier_needs_one_image() -> None:
    assert _plan({}, _archive("environment/Dockerfile")) == [
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None)
    ]


def test_separate_verifier_builds_from_tests() -> None:
    config = {"verifier": {"environment_mode": "separate"}}

    assert _plan(config, _archive("environment/Dockerfile", "tests/Dockerfile")) == [
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("verifier", "dockerfile", "sort-task/tests", "Dockerfile", None),
    ]


def test_separate_verifier_without_its_own_environment_reuses_a_prebuilt_main_image() -> None:
    """Harbor 0.20 copies ``[environment]`` for the verifier, so a named main image wins over tests/."""
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}, "verifier": {"environment_mode": "separate"}}

    assert _plan(config, _archive("tests/Dockerfile"), ["ghcr.io"]) == [
        ("main", "prebuilt", None, None, "ghcr.io/org/sort:1"),
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


# Harbor starts every Docker task from its own compose file, whose `main` is built from `environment/` with
# `Dockerfile`, or is the task's `docker_image`, and lays the task's `docker-compose.yaml` over it. The task's file
# adds services and adjusts `main`; it isn't the whole list.


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
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("service/db", "dockerfile", "sort-task/environment/db", "Dockerfile.db", None),
        ("service/cache", "prebuilt", None, None, "docker.io/library/redis:7"),
    ]


def test_a_compose_main_that_only_adjusts_main_is_built_from_the_environment_dockerfile() -> None:
    compose = "services:\n  main:\n    depends_on: [db]\n  db:\n    build: ./db\n"
    archive = _archive("environment/Dockerfile", "environment/db/Dockerfile", compose=compose)

    assert _plan({}, archive) == [
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("service/db", "dockerfile", "sort-task/environment/db", "Dockerfile", None),
    ]


def test_a_compose_file_without_main_still_builds_main() -> None:
    compose = "services:\n  cache:\n    image: redis:7\n"

    assert _plan({}, _archive("environment/Dockerfile", compose=compose), ["docker.io"]) == [
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None),
        ("service/cache", "prebuilt", None, None, "docker.io/library/redis:7"),
    ]


def test_a_compose_file_with_no_services_builds_main() -> None:
    assert _plan({}, _archive("environment/Dockerfile", compose="services: {}\n")) == [
        ("main", "dockerfile", "sort-task/environment", "Dockerfile", None),
    ]


def test_a_docker_image_task_still_plans_its_compose_services() -> None:
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}}
    compose = "services:\n  main:\n    depends_on: [cache]\n  cache:\n    image: redis:7\n"

    assert _plan(config, _archive(compose=compose), ["ghcr.io", "docker.io"]) == [
        ("main", "prebuilt", None, None, "ghcr.io/org/sort:1"),
        ("service/cache", "prebuilt", None, None, "docker.io/library/redis:7"),
    ]


def test_the_planner_plans_the_images_the_parser_resolves(tmp_path: Path) -> None:
    """Interpolation and `.env` apply to a service's image, so the planner and the run agree on it."""
    environment = tmp_path / "sort-task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / ".env").write_text("REGISTRY=mirror.example.com\n", encoding="utf-8")
    compose = "services:\n  cache:\n    image: ${REGISTRY:-docker.io}/library/redis:7\n"
    (environment / "docker-compose.yaml").write_text(compose, encoding="utf-8")
    parsed = parse_compose(tmp_path / "sort-task", read_entrypoint=False)
    archive = TaskArchive(
        root="sort-task",
        files=frozenset({"sort-task/environment/Dockerfile", "sort-task/environment/docker-compose.yaml"}),
        compose=compose,
        compose_task=parsed,
    )

    images = plan_images({}, archive, allowed_registries=[])

    assert [(image.type, image.image) for image in images[1:]] == [
        ("service/cache", "mirror.example.com/library/redis:7")
    ]
    assert parsed is not None and [image.type for image in images[1:]] == [
        f"service/{name}" for name in parsed.services
    ]


def test_a_compose_file_that_was_not_parsed_is_refused() -> None:
    archive = _archive("environment/Dockerfile", compose="services:\n  cache:\n    image: redis:7\n")

    with pytest.raises(ImagePlanError) as error:
        plan_images({}, archive, allowed_registries=[])

    assert error.value.code == "invalid_task_archive"


@pytest.mark.parametrize(
    ("build", "context_path", "dockerfile"),
    [
        ("{dockerfile: Dockerfile.agent}", "sort-task/environment", "Dockerfile.agent"),
        ("{context: ., dockerfile: Dockerfile.agent}", "sort-task/environment", "Dockerfile.agent"),
        ("./app", "sort-task/environment/app", "Dockerfile"),
    ],
)
def test_a_compose_main_build_overrides_the_environment_dockerfile(
    build: str, context_path: str, dockerfile: str
) -> None:
    """Compose merges `main.build` over the base's, so a field it leaves out keeps the base's value."""
    files = ("environment/Dockerfile.agent", "environment/app/Dockerfile")
    archive = _archive(*files, compose=f"services:\n  main:\n    build: {build}\n")

    assert _plan({}, archive) == [("main", "dockerfile", context_path, dockerfile, None)]


def test_a_compose_main_image_only_names_main_when_main_is_built() -> None:
    """Over the base's `build` and `pull_policy: build`, compose builds `main` and tags it with this name."""
    archive = _archive("environment/Dockerfile", compose="services:\n  main:\n    image: sort-env:dev\n")

    assert _plan({}, archive) == [("main", "dockerfile", "sort-task/environment", "Dockerfile", None)]


def test_a_compose_main_image_replaces_the_docker_image() -> None:
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}}
    archive = _archive(compose="services:\n  main:\n    image: ghcr.io/org/sort:2\n")

    assert _plan(config, archive, ["ghcr.io"]) == [("main", "prebuilt", None, None, "ghcr.io/org/sort:2")]


def test_a_docker_image_wins_over_a_compose_main_build() -> None:
    """Harbor builds over a `docker_image` only with `force_build`, which a trial never sets."""
    config = {"environment": {"docker_image": "ghcr.io/org/sort:1"}}
    compose = "services:\n  main:\n    build: {context: ., dockerfile: Dockerfile.custom}\n"
    archive = _archive("environment/Dockerfile", "environment/Dockerfile.custom", compose=compose)

    assert _plan(config, archive, ["ghcr.io"]) == [("main", "prebuilt", None, None, "ghcr.io/org/sort:1")]


def test_a_compose_service_named_verifier_runs_next_to_a_separate_verifier() -> None:
    """kumo's game server is a service called `verifier`; its image's type can't be mistaken for the verifier's."""
    config = {"verifier": {"environment_mode": "separate"}}
    compose = "services:\n  verifier:\n    image: redis:7\n"

    images = _plan_images(config, _archive("environment/Dockerfile", "tests/Dockerfile", compose=compose), [])

    assert [image.type for image in images] == ["main", "service/verifier", "verifier"]


def test_a_compose_service_named_verifier_is_planned_without_a_separate_verifier() -> None:
    compose = "services:\n  verifier:\n    image: redis:7\n"

    images = _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert [image.type for image in images] == ["main", "service/verifier"]


@pytest.mark.parametrize("service", ["main", "cache"])
@pytest.mark.parametrize("option", ["target: test", "args: {MODE: fast}", "no_cache: true"])
def test_a_compose_build_option_builder_cannot_apply_fails_planning(service: str, option: str) -> None:
    """Builder builds only a context and a Dockerfile; dropping any other option would build a different image."""
    compose = f"services:\n  {service}:\n    build: {{context: ., {option}}}\n"

    with pytest.raises(ImagePlanError) as error:
        _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert error.value.code == "unsupported_compose_build_option"
    assert option.split(":")[0] in str(error.value)


def test_a_compose_build_option_the_parser_does_not_know_fails_before_planning() -> None:
    """The parser refuses build options it doesn't read, such as `dockerfile_inline`, naming the key."""
    compose = "services:\n  cache:\n    build: {context: ., dockerfile_inline: FROM scratch}\n"

    with pytest.raises(ImagePlanError) as error:
        _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert error.value.code == "invalid_compose_file"
    assert "services.cache.build.dockerfile_inline" in str(error.value)


@pytest.mark.parametrize(
    ("config", "archive", "code"),
    [
        ({"environment": {"os": "windows"}}, _archive("environment/Dockerfile"), "windows_unsupported"),
        ({}, _archive("instruction.md"), "main_dockerfile_missing"),
        ({}, _archive(compose="services:\n  cache:\n    image: redis:7\n"), "main_dockerfile_missing"),
        (
            {"verifier": {"environment_mode": "separate"}},
            _archive("environment/Dockerfile"),
            "verifier_dockerfile_missing",
        ),
        ({"environment": {"docker_image": "quay.io/org/x:1"}}, _archive(), "registry_not_allowed"),
        ({}, _archive(compose="services:\n  main:\n    build: ../../elsewhere\n"), "invalid_compose_file"),
        ({}, _archive(compose="services: []\n"), "invalid_compose_file"),
        ({}, _archive(compose="services:\n  main:\n    build: ./missing\n"), "dockerfile_missing"),
        (
            {},
            _archive("Dockerfile", compose="services:\n  main:\n    build: {context: ., dockerfile: ../Dockerfile}\n"),
            "invalid_compose_file",
        ),
        (
            {},
            _archive(compose="services:\n  main:\n    build: {context: ., dockerfile: /etc/Dockerfile}\n"),
            "invalid_compose_file",
        ),
        ({"environment": {"cpus": "lots"}}, _archive("environment/Dockerfile"), "invalid_task_config"),
    ],
)
def test_unbuildable_tasks_fail_with_a_stable_code(config: dict, archive: TaskArchive, code: str) -> None:
    with pytest.raises(ImagePlanError) as error:
        _plan_images(config, archive, ["docker.io"])

    assert error.value.code == code


def test_without_allowed_registries_any_registry_is_left_to_builder() -> None:
    """Builder holds the deployment's allowlist; a caller's list only narrows it."""
    config = {"environment": {"docker_image": "quay.io/org/x:1"}}

    assert _plan(config, _archive()) == [("main", "prebuilt", None, None, "quay.io/org/x:1")]


@pytest.mark.parametrize("allowed", [["docker.io"], ["index.docker.io"], ["registry-1.docker.io"]])
def test_docker_hub_shorthand_matches_any_spelling_of_docker_hub(allowed: list[str]) -> None:
    config = {"environment": {"docker_image": "ubuntu:24.04"}}

    assert _plan(config, _archive(), allowed)[0][1] == "prebuilt"


def test_an_image_that_is_not_a_reference_fails_planning() -> None:
    with pytest.raises(ImagePlanError) as error:
        _plan_images({"environment": {"docker_image": "Not A Ref"}}, _archive(), [])

    assert error.value.code == "invalid_image_reference"


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


@pytest.mark.parametrize(
    ("dockerfile", "pin"),
    [
        ("FROM python:3.12\n", None),
        ("FROM --platform=linux/amd64 python:3.12\n", "linux/amd64"),
        ("FROM --platform=linux/x86_64 python:3.12\n", "linux/amd64"),
        ("from --platform=LINUX/ARM64/v8 python:3.12 AS app\n", "linux/arm64"),
        ("FROM --platform=$BUILDPLATFORM golang AS build\nFROM --platform=linux/arm64 alpine\n", "linux/arm64"),
        ("FROM --platform=linux/amd64 golang AS build\nFROM alpine\n", None),
        ("FROM --platform=$TARGETPLATFORM alpine\n", None),
        ("FROM \\\n  --platform=linux/amd64 \\\n  ubuntu\n", "linux/amd64"),
        ("# FROM --platform=linux/amd64 ubuntu\nFROM ubuntu\n", None),
        # A comment line never continues, even ending in a backslash, and is dropped from an instruction it splits.
        ("# a note \\\nFROM --platform=linux/arm64 alpine\n", "linux/arm64"),
        ("FROM \\\n# a note\n  --platform=linux/amd64 ubuntu\n", "linux/amd64"),
    ],
)
def test_a_dockerfile_pins_the_platform_its_final_stage_names_plainly(dockerfile: str, pin: str | None) -> None:
    assert dockerfile_pin(dockerfile) == pin


def test_a_pinned_dockerfile_in_the_archive_pins_its_image(tmp_path: Path) -> None:
    path = tmp_path / "task_archive"
    _write_tar(path, {"sort-task/environment/Dockerfile": b"FROM --platform=linux/arm64 python:3.12\n"})

    (image,) = plan_images({}, TaskArchive.read(path), allowed_registries=[])

    assert image.platform == "linux/arm64"


def test_compose_services_pin_the_platform_they_name_plainly() -> None:
    compose = """
services:
  main:
    build: .
    platform: linux/amd64
  cache:
    image: redis:7
    platform: ${CACHE_PLATFORM}
  db:
    image: postgres:16
    platform: linux/arm64/v8
"""
    images = _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert [(image.type, image.platform) for image in images] == [
        ("main", "linux/amd64"),
        ("service/cache", None),
        ("service/db", "linux/arm64"),
    ]


def test_a_compose_service_platform_set_in_dot_env_is_pinned(tmp_path: Path) -> None:
    """Planning pins the platform the parser resolved, with `.env`'s value filled in as Compose fills it."""
    compose = "services:\n  cache:\n    image: redis:7\n    platform: ${TARGET_PLATFORM}\n"
    environment = tmp_path / "sort-task" / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(compose, encoding="utf-8")
    (environment / ".env").write_text("TARGET_PLATFORM=linux/arm64\n", encoding="utf-8")
    archive = replace(
        _archive("environment/Dockerfile", compose=compose),
        compose_task=parse_compose(tmp_path / "sort-task", read_entrypoint=False),
    )

    images = plan_images({}, archive, allowed_registries=[])

    assert {image.type: image.platform for image in images}["service/cache"] == "linux/arm64"


def test_a_compose_service_naming_an_unreadable_platform_is_refused() -> None:
    compose = "services:\n  cache:\n    image: redis:7\n    platform: arm64\n"

    with pytest.raises(ImagePlanError) as error:
        _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert error.value.code == "invalid_compose_file"


def test_a_compose_main_platform_pins_main_built_from_the_environment() -> None:
    """Compose merges `main.platform` over Harbor's base, so it pins `main` even when only adjusting it."""
    compose = "services:\n  main:\n    depends_on: [db]\n    platform: linux/amd64\n  db:\n    build: ./db\n"
    archive = _archive("environment/Dockerfile", "environment/db/Dockerfile", compose=compose)

    images = _plan_images({}, archive, [])

    assert [(image.type, image.platform) for image in images] == [("main", "linux/amd64"), ("service/db", None)]


def test_a_compose_service_pins_an_architecture_alias_as_buildkit_names_it() -> None:
    """BuildKit takes `x86_64` and `aarch64` for `amd64` and `arm64`; a task that names one still builds."""
    compose = "services:\n  main:\n    build: .\n    platform: linux/x86_64\n  cache:\n    image: redis:7\n    platform: linux/aarch64\n"
    images = _plan_images({}, _archive("environment/Dockerfile", compose=compose), [])

    assert [(image.type, image.platform) for image in images] == [
        ("main", "linux/amd64"),
        ("service/cache", "linux/arm64"),
    ]
