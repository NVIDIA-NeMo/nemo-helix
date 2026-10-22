# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor task's compose file as ``ComposeServices``: detection, merge, interpolation, mapping and rejections.

Every test writes its own task directory under ``tmp_path``. The last test sweeps a local checkout of the
Terminal-Bench tasks and runs only when ``TB3_TASKS_DIR`` points at one.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
from nemo_evals.harbor.compose.images import dockerfile_entrypoint, normalize_image
from nemo_evals.harbor.compose.models import ComposeBuild, ComposeError, ComposeImage, ComposeProblems, ComposeTask
from nemo_evals.harbor.compose.parser import parse_compose
from nemo_evals.harbor.compose.resolve import Interpolator, merge_documents
from nemo_evals.harbor.compose.spec import compose_provenance, compose_services_kwarg, to_compose_services
from nhx_sandbox.harbor_opensandbox_services import ComposeServices

DOCKERFILE = "FROM python:3.12-slim\n"


def _task(
    root: Path,
    compose: str | None,
    *,
    task_toml: str = "",
    dockerfile: str | None = DOCKERFILE,
    files: dict[str, str] | None = None,
    compose_name: str = "docker-compose.yaml",
) -> Path:
    """A task directory named ``demo`` with the given compose file and any extra files under ``environment/``."""
    task = root / "demo"
    environment = task / "environment"
    environment.mkdir(parents=True)

    if compose is not None:
        (environment / compose_name).write_text(textwrap.dedent(compose), encoding="utf-8")
    if dockerfile is not None:
        (environment / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    if task_toml:
        (task / "task.toml").write_text(textwrap.dedent(task_toml), encoding="utf-8")

    for name, content in (files or {}).items():
        (environment / name).parent.mkdir(parents=True, exist_ok=True)
        (environment / name).write_text(content, encoding="utf-8")
    return task


def _spec(tmp_path: Path, compose: str, images: dict[str, str] | None = None, **kwargs: Any) -> dict[str, Any]:
    """The ``compose_services`` kwarg for a compose file, as the dispatcher would pass it."""
    result = compose_services_kwarg(_task(tmp_path, compose, **kwargs), images)
    assert result is not None
    return result


def _service(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """The service called ``name`` in a ``compose_services`` spec."""
    return next(service for service in spec["services"] if service["name"] == name)


def _error(tmp_path: Path, compose: str | None, **kwargs: Any) -> ComposeError:
    """The error ``parse_compose`` raises for a task with this compose file."""
    with pytest.raises(ComposeError) as caught:
        parse_compose(_task(tmp_path, compose, **kwargs))
    return caught.value


# ---------------------------------------------------------------------------------------------------------------------
# Images


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        ("postgres:18", "docker.io/library/postgres:18"),
        ("apache/kafka-native:4.3.1", "docker.io/apache/kafka-native:4.3.1"),
        ("redis", "docker.io/library/redis:latest"),
        ("ghcr.io/org/img:1", "ghcr.io/org/img:1"),
        ("localhost:5000/img:1", "localhost:5000/img:1"),
        ("index.docker.io/library/redis:7", "docker.io/library/redis:7"),
        ("postgres@sha256:" + "a" * 64, "docker.io/library/postgres@sha256:" + "a" * 64),
        ("redis:7-alpine@sha256:" + "b" * 64, "docker.io/library/redis:7-alpine@sha256:" + "b" * 64),
    ],
)
def test_image_names_are_normalized_as_docker_does(written: str, normalized: str) -> None:
    assert normalize_image(written) == normalized


def test_invalid_image_name_is_a_typed_error() -> None:
    with pytest.raises(ComposeError) as caught:
        normalize_image("not a ref")
    assert caught.value.code == "compose_invalid"


# ---------------------------------------------------------------------------------------------------------------------
# Interpolation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("$$(cat /x)", "$(cat /x)"),
        ("$$$$", "$$"),
        ("${A}-$A", "one-one"),
        ("${MISSING:-fallback}", "fallback"),
        ("${EMPTY:-fallback}", "fallback"),
        ("${EMPTY-kept}", ""),
        ("${MISSING-fallback}", "fallback"),
        ("${MISSING:-${A}}", "one"),
        ("${MISSING}/$MISSING", "/"),
        ("${A:+alt}", "alt"),
        ("${EMPTY:+alt}", ""),
        ("${EMPTY+alt}", "alt"),
        ("${MISSING+alt}", ""),
        ("${A:?needed}", "one"),
        ("${EMPTY?needed}", ""),
        ("[ $? -eq 28 ]", "[ $? -eq 28 ]"),
        ("cost: $5", "cost: $5"),
        ("end$", "end$"),
    ],
)
def test_interpolation_follows_compose(text: str, expected: str) -> None:
    problems = ComposeProblems()
    assert Interpolator({"A": "one", "EMPTY": ""}, problems).string(text, "x") == expected
    problems.raise_if_any("compose.yaml")


@pytest.mark.parametrize("text", ["${NOPE:?set NOPE}", "${EMPTY:?set EMPTY}", "${NOPE?set NOPE}"])
def test_a_required_variable_that_is_missing_is_invalid(text: str) -> None:
    problems = ComposeProblems()
    Interpolator({"EMPTY": ""}, problems).string(text, "services.a.environment.X")

    with pytest.raises(ComposeError) as caught:
        problems.raise_if_any("compose.yaml")

    assert caught.value.code == "compose_invalid"
    assert "services.a.environment.X:" in caught.value.detail
    assert "is required: set" in caught.value.detail


def test_an_invalid_value_does_not_hide_unsupported_features() -> None:
    problems = ComposeProblems()
    problems.unsupported("services.a.privileged")
    problems.invalid("services.b.ports[0]", "expected a port")

    with pytest.raises(ComposeError) as caught:
        problems.raise_if_any("compose.yaml")

    assert caught.value.code == "compose_invalid"
    assert "services.b.ports[0]: expected a port" in caught.value.detail
    assert caught.value.paths == ("services.a.privileged",)


def test_unset_variable_is_empty_as_under_compose(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          db:
            image: postgres:18
            environment:
              TOKEN: ${HOST_TOKEN}
              TEST_DIR: ${TEST_DIR}
        """,
    )
    assert _service(spec, "db")["env"] == {"TOKEN": "", "TEST_DIR": ""}


def test_task_toml_env_and_harbor_infra_variables_interpolate(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            environment:
              DB_HOST: db
              CPUS: ${CPUS}
              MEM: ${MEMORY}
          db:
            image: postgres:${PG_VERSION}
            environment:
              PASSWORD: ${DB_PASSWORD:-fallback}
        """,
        task_toml="""
        [environment]
        cpus = 2
        memory_mb = 4096
        env = { PG_VERSION = "17", DB_PASSWORD = "from-task" }
        """,
    )

    assert _service(spec, "db")["image"] == "docker.io/library/postgres:17"
    assert _service(spec, "db")["env"] == {"PASSWORD": "from-task"}
    assert spec["main"]["env"] == {"DB_HOST": "db", "CPUS": "2", "MEM": "4096M"}


REGISTRY_COMPOSE = "services:\n  cache:\n    image: ${REGISTRY:-docker.io}/library/redis:7\n"


def test_environment_dotenv_file_interpolates_as_compose_reads_it(tmp_path: Path) -> None:
    spec = _spec(tmp_path, REGISTRY_COMPOSE, files={".env": "# registry\nREGISTRY=mirror.example.com\n"})
    assert _service(spec, "cache")["image"] == "mirror.example.com/library/redis:7"


def test_dotenv_has_the_lowest_priority(tmp_path: Path) -> None:
    compose = (
        "services:\n  cache:\n    image: ${REGISTRY}/redis:${TAG}\n    environment:\n      HOME_DIR: ${CONTEXT_DIR}\n"
    )
    spec = _spec(
        tmp_path,
        compose,
        task_toml='[environment]\nenv = { REGISTRY = "task.example.com", CONTEXT_DIR = "ignored" }\n',
        files={".env": "REGISTRY=dotenv.example.com\nTAG=7\nCONTEXT_DIR=/elsewhere\n"},
    )
    cache = _service(spec, "cache")
    assert cache["image"] == "task.example.com/redis:7"
    assert cache["env"] == {"HOME_DIR": "."}


def test_the_hosts_environment_is_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGISTRY", "host.example.com")
    assert _service(_spec(tmp_path, REGISTRY_COMPOSE), "cache")["image"] == "docker.io/library/redis:7"


def test_dotenv_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, REGISTRY_COMPOSE)
    target = tmp_path / "outside.env"
    target.write_text("REGISTRY=leak.example.com\n", encoding="utf-8")
    (task / "environment" / ".env").symlink_to(target)
    with pytest.raises(ComposeError, match="symlink"):
        parse_compose(task)


# ---------------------------------------------------------------------------------------------------------------------
# Detection


def test_task_without_a_compose_file_is_single_container(tmp_path: Path) -> None:
    task = _task(tmp_path, None)
    assert parse_compose(task) is None
    assert compose_services_kwarg(task) is None


@pytest.mark.parametrize("name", ["docker-compose.yml", "compose.yaml", "compose.yml"])
def test_compose_file_with_a_name_harbor_ignores_is_rejected_with_a_rename_hint(tmp_path: Path, name: str) -> None:
    error = _error(tmp_path, "services: {}\n", compose_name=name)
    assert error.code == "compose_unsupported_name"
    assert "rename" in error.detail
    assert name in (error.compose_path or "")


@pytest.mark.parametrize(
    "compose",
    ["- just\n- a list\n", "just a string\n", "version: '3'\n", "services: [a, b]\n", "services: {a: [1]}\n", "a: [\n"],
)
def test_malformed_compose_file_is_invalid(tmp_path: Path, compose: str) -> None:
    assert _error(tmp_path, compose).code == "compose_invalid"


def test_empty_compose_file_is_invalid(tmp_path: Path) -> None:
    assert _error(tmp_path, "").code == "compose_invalid"


def test_oversized_compose_file_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: x\n" + "# padding\n" * 30_000)
    assert error.code == "compose_invalid"
    assert "limit" in error.detail


def test_file_that_only_overrides_main_parses_but_has_no_services_to_run(tmp_path: Path) -> None:
    compose = "services:\n  main:\n    environment:\n      A: b\n"
    task = _parsed(tmp_path, compose)
    assert task.services == {} and task.main.env == {"A": "b"}
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_override_only"
    with pytest.raises(ComposeError, match="compose_override_only"):
        compose_services_kwarg(tmp_path / "demo")


def test_compose_file_is_never_executed(tmp_path: Path) -> None:
    error = _error(tmp_path, "services: !!python/object/apply:os.system ['echo hi']\n")
    assert error.code == "compose_invalid"


# ---------------------------------------------------------------------------------------------------------------------
# Merge with Harbor's base file


def _parsed(tmp_path: Path, compose: str, **kwargs: Any) -> ComposeTask:
    """The ``ComposeTask`` ``parse_compose`` reads from a task with this compose file."""
    task = parse_compose(_task(tmp_path, compose, **kwargs))
    assert task is not None
    return task


def test_main_is_built_from_environment_by_default(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  db:\n    image: redis\n")
    assert task.main_dockerfile == "Dockerfile"
    assert task.compose_path == "demo/environment/docker-compose.yaml"


def test_task_file_chooses_mains_dockerfile(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main:
            build:
              dockerfile: Dockerfile.agent
          db:
            image: redis
        """,
        files={"Dockerfile.agent": 'FROM scratch\nENTRYPOINT ["/agent"]\n'},
    )

    assert task.main_dockerfile == "Dockerfile.agent"
    assert task.main.entrypoint == ["/agent"]


def test_mains_build_context_and_dockerfile_combine(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main:
            build:
              context: ./agent
              dockerfile: docker/Main.Dockerfile
          db:
            image: redis
        """,
        files={"agent/docker/Main.Dockerfile": "FROM scratch\n"},
    )

    assert task.main_dockerfile == "agent/docker/Main.Dockerfile"


def test_prebuilt_task_ignores_mains_build_and_needs_no_dockerfile(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main:
            build: {dockerfile: Dockerfile.agent}
          db:
            image: redis
        """,
        task_toml='[environment]\ndocker_image = "registry.example.com/demo/main:1"\n',
        dockerfile=None,
    )

    assert task.main_dockerfile is None
    assert task.main.entrypoint is None


def test_main_build_and_image_are_exposed_for_image_planning(tmp_path: Path) -> None:
    built = _parsed(
        tmp_path / "built",
        "services:\n  main:\n    build: {context: ./app, dockerfile: Dockerfile.main}\n  db:\n    image: redis\n",
        files={"app/Dockerfile.main": DOCKERFILE},
    )
    assert built.main_build == ComposeBuild(context_path="environment/app", dockerfile="Dockerfile.main")
    assert built.main_image is None

    prebuilt = _parsed(
        tmp_path / "prebuilt",
        "services:\n  db:\n    image: redis\n",
        task_toml='[environment]\ndocker_image = "registry.example.com/demo/main:1"\n',
        dockerfile=None,
    )
    assert prebuilt.main_build is None
    assert prebuilt.main_image == "registry.example.com/demo/main:1"

    replaced = _parsed(
        tmp_path / "replaced",
        "services:\n  main:\n    image: ghcr.io/org/sort:2\n  db:\n    image: redis\n",
        task_toml='[environment]\ndocker_image = "registry.example.com/demo/main:1"\n',
        dockerfile=None,
    )
    assert replaced.main_image == "ghcr.io/org/sort:2"


def test_main_dockerfile_is_left_unread_when_only_images_are_planned(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n", dockerfile=None)
    with pytest.raises(ComposeError, match="main's Dockerfile Dockerfile is not in the task"):
        parse_compose(task)
    planned = parse_compose(task, read_entrypoint=False)
    assert planned is not None and planned.main_build == ComposeBuild(
        context_path="environment", dockerfile="Dockerfile"
    )
    assert planned.main.entrypoint is None


def test_mains_own_command_is_rejected_because_the_sandbox_keeps_main_alive(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            command: ["python", "app.py"]
          db:
            image: redis
        """,
    )
    assert error.code == "compose_unsupported"
    assert error.paths == ("services.main.command",)


def test_missing_mains_dockerfile_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  db:\n    image: redis\n", dockerfile=None)
    assert error.code == "compose_invalid"
    assert "Dockerfile" in error.detail


@pytest.mark.parametrize(
    ("merge_base", "merge_over", "expected"),
    [
        ({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}, {"a": {"x": 1, "y": 3}}),
        ({"expose": ["1"]}, {"expose": ["1", "2"]}, {"expose": ["1", "2"]}),
        ({"cap_add": ["A"]}, {"cap_add": ["B"]}, {"cap_add": ["A", "B"]}),
        ({"command": ["a"]}, {"command": ["b"]}, {"command": ["b"]}),
    ],
)
def test_merge_follows_compose_rules(merge_base: dict, merge_over: dict, expected: dict) -> None:
    assert merge_documents(merge_base, merge_over) == expected


# ---------------------------------------------------------------------------------------------------------------------
# Mapping


def test_services_are_mapped_with_roles_and_start_order(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            depends_on:
              seeder: {condition: service_completed_successfully}
              kafka: {condition: service_healthy}
          seeder:
            image: acme/seeder:1
            depends_on:
              kafka: {condition: service_healthy}
          kafka:
            image: apache/kafka:3
            healthcheck:
              test: ["CMD-SHELL", "nc -z localhost 9092"]
          loadgen:
            image: acme/loadgen:1
            depends_on:
              main: {condition: service_healthy}
          other:
            image: redis
        """,
    )

    assert [(service["name"], service.get("role", "sidecar")) for service in spec["services"]] == [
        ("kafka", "sidecar"),
        ("seeder", "run_once"),
        ("loadgen", "after_main"),
        ("other", "sidecar"),
    ]
    assert "readiness" not in _service(spec, "seeder")
    assert _service(spec, "kafka")["readiness"] == {"exec": ["sh", "-c", "nc -z localhost 9092"]}


def test_service_depended_on_for_completion_by_another_service_is_run_once(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          migrate:
            image: acme/migrate:1
          app:
            image: acme/app:1
            depends_on:
              migrate: {condition: service_completed_successfully}
        """,
    )

    assert _service(spec, "migrate")["role"] == "run_once"
    assert "role" not in _service(spec, "app")


def test_depends_on_list_form_means_started(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main:
            depends_on: [db]
          db:
            image: redis
        """,
    )
    assert task.main.depends_on == {"db": "service_started"}


def test_dependency_cycle_is_invalid(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          a: {image: x, depends_on: [b]}
          b: {image: x, depends_on: [a]}
        """,
    )
    assert error.code == "compose_invalid"
    assert set(error.paths) == {"a", "b"}


def test_dependency_on_an_unknown_service_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    depends_on: [ghost]\n")
    assert error.code == "compose_invalid"
    assert "ghost" in error.detail


def test_service_and_main_waiting_on_each_other_is_invalid(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            depends_on: [a]
          a:
            image: redis
            depends_on: [main]
        """,
    )
    assert error.code == "compose_invalid"


@pytest.mark.parametrize(
    ("healthcheck", "readiness"),
    [
        ({"test": ["CMD", "curl", "-f", "http://localhost"]}, {"exec": ["curl", "-f", "http://localhost"]}),
        ({"test": ["CMD-SHELL", "pg_isready -U x"]}, {"exec": ["sh", "-c", "pg_isready -U x"]}),
        ({"test": "pg_isready -U x"}, {"exec": ["sh", "-c", "pg_isready -U x"]}),
        ({"test": ["NONE"]}, None),
        ({"disable": True}, None),
        ({"interval": "5s"}, None),
    ],
)
def test_healthcheck_becomes_readiness(tmp_path: Path, healthcheck: dict, readiness: dict | None) -> None:
    compose = {"services": {"db": {"image": "redis", "healthcheck": healthcheck}}}
    spec = compose_services_kwarg(_task(tmp_path, json.dumps(compose)))
    assert spec is not None
    assert _service(spec, "db").get("readiness") == readiness


def test_entrypoint_and_command_become_command_and_args(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          db:
            image: mysql:8
            entrypoint: ["/entry.sh"]
            command:
              - mysqld
              - --server-id=1
          web:
            image: nginx
            command: nginx -g "daemon off;"
        """,
    )

    assert _service(spec, "db")["command"] == ["/entry.sh"]
    assert _service(spec, "db")["args"] == ["mysqld", "--server-id=1"]
    assert _service(spec, "web")["args"] == ["nginx", "-g", "daemon off;"]
    assert "command" not in _service(spec, "web")


def test_environment_forms_and_scalar_values(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          a:
            image: redis
            environment:
              PORT: 8080
              DEBUG: true
              RATIO: 0.5
              NAME: "x y"
          b:
            image: redis
            environment:
              - K=v=w
              - EMPTY=
        """,
    )

    assert _service(spec, "a")["env"] == {"PORT": "8080", "DEBUG": "true", "RATIO": "0.5", "NAME": "x y"}
    assert _service(spec, "b")["env"] == {"K": "v=w", "EMPTY": ""}


def test_environment_variable_without_a_value_is_left_out(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "services:\n  a:\n    image: redis\n    environment:\n      - FROM_HOST\n      - KEPT=1\n")
    assert _service(spec, "a")["env"] == {"KEPT": "1"}


def test_env_file_is_read_from_the_archive_and_environment_wins(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            env_file: main.env
          a:
            image: redis
            env_file: [a.env]
            environment:
              X: from-compose
        """,
        files={
            "a.env": "# comment\nX=from-file\nY='quoted'\nexport Z=3\n",
            "main.env": "M=1\n",
        },
    )

    assert _service(spec, "a")["env"] == {"X": "from-compose", "Y": "quoted", "Z": "3"}
    assert spec["main"]["env"] == {"M": "1"}


def test_env_file_outside_environment_dir_is_rejected(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    env_file: ../secret.env\n")
    assert error.code == "compose_unsupported"


def test_missing_required_env_file_is_invalid_but_optional_is_fine(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    env_file: nope.env\n")
    assert error.code == "compose_invalid"
    spec = _spec(
        tmp_path / "second",
        "services:\n  a:\n    image: redis\n    env_file:\n      - {path: nope.env, required: false}\n",
    )
    assert "env" not in _service(spec, "a")


def test_ports_come_from_expose_and_the_container_side_of_ports(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          a:
            image: redis
            expose: ["6379", 6380]
            ports:
              - "8080:80"
              - "127.0.0.1:9090:90/tcp"
              - 7000
              - {target: 7100, published: 7101}
              - "6379:6379"
        """,
    )

    assert _service(spec, "a")["ports"] == [6379, 6380, 80, 90, 7000, 7100]


def test_port_ranges_are_rejected(tmp_path: Path) -> None:
    error = _error(tmp_path, 'services:\n  a:\n    image: redis\n    ports: ["8000-8005:8000-8005"]\n')
    assert error.code == "compose_unsupported"


def test_named_volumes_and_mounts(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            volumes:
              - shared:/data:ro
          a:
            image: redis
            volumes:
              - shared:/data
              - {type: volume, source: cache, target: /cache, read_only: true}
        volumes:
          shared: {}
          cache:
          unused:
        """,
    )

    assert spec["volumes"] == ["shared", "cache"]
    assert spec["main"]["volume_mounts"] == [{"name": "shared", "mount_path": "/data", "read_only": True}]
    assert _service(spec, "a")["volume_mounts"] == [
        {"name": "shared", "mount_path": "/data"},
        {"name": "cache", "mount_path": "/cache", "read_only": True},
    ]


def test_volume_names_become_names_a_pod_accepts(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            volumes: [agent1_out:/shared/agent1]
          agent1:
            image: redis
            volumes: [agent1_out:/agent_output, Cache.V2:/cache]
        volumes:
          agent1_out:
          Cache.V2:
        """,
    )
    assert spec["volumes"] == ["agent1-out", "cache-v2"]
    assert spec["main"]["volume_mounts"] == [{"name": "agent1-out", "mount_path": "/shared/agent1"}]
    assert [mount["name"] for mount in _service(spec, "agent1")["volume_mounts"]] == ["agent1-out", "cache-v2"]


def test_volumes_whose_pod_names_would_clash_are_invalid(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          a:
            image: redis
            volumes: [data_1:/one, data.1:/two]
        volumes:
          data_1:
          data.1:
        """,
    )
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_invalid"
    assert caught.value.paths == ("data_1", "data.1")


def test_harbor_log_mounts_on_main_need_nothing(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            volumes:
              - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
              - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
          db:
            image: redis
        """,
    )
    assert "volume_mounts" not in spec.get("main", {})
    assert "volumes" not in spec


def test_a_harbor_log_mount_on_a_service_is_shared_with_main(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            volumes:
              - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
          game:
            image: redis
            volumes:
              - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
        """,
    )
    assert spec["volumes"] == ["harbor-verifier-logs"]
    assert _service(spec, "game")["volume_mounts"] == [{"name": "harbor-verifier-logs", "mount_path": "/logs/verifier"}]
    assert spec["main"]["volume_mounts"] == [{"name": "harbor-verifier-logs", "mount_path": "/logs/verifier"}]


@pytest.mark.parametrize(
    "volume",
    [
        "$HOST_AGENT_LOGS_PATH:$ENV_AGENT_LOGS_PATH:ro",
        {"type": "bind", "source": "${HOST_AGENT_LOGS_PATH}", "target": "${ENV_AGENT_LOGS_PATH}", "read_only": True},
    ],
)
def test_harbor_log_mounts_are_recognized_in_every_spelling(tmp_path: Path, volume: Any) -> None:
    task = _parsed(tmp_path, json.dumps({"services": {"game": {"image": "redis", "volumes": [volume]}}}))
    assert task.services["game"].volume_mounts[0].model_dump() == {
        "name": "harbor-agent-logs",
        "mount_path": "/logs/agent",
        "read_only": True,
    }


def test_only_harbor_log_mounts_written_exactly_are_recognized(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    volumes: ['${HOST_AGENT_LOGS_PATH}:/elsewhere']\n")
    assert error.code == "compose_unsupported"
    assert any(path.startswith("services.a.volumes[0]") for path in error.paths)


def test_a_volume_named_like_a_harbor_log_volume_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\nvolumes:\n  harbor-agent-logs: {}\n")
    assert error.code == "compose_invalid"
    assert "volumes.harbor-agent-logs: this name is reserved" in error.detail


def test_harbor_log_directory_variables_fill_in(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "services:\n  a:\n    image: redis\n    environment: {LOGS: '${ENV_VERIFIER_LOGS_PATH}'}\n")
    assert _service(spec, "a")["env"] == {"LOGS": "/logs/verifier"}


@pytest.mark.parametrize(
    "volume",
    ["./data:/data", "/host:/data", "~/x:/data", "/anonymous", {"type": "bind", "source": "/h", "target": "/t"}],
)
def test_bind_mounts_and_anonymous_volumes_are_rejected(tmp_path: Path, volume: Any) -> None:
    compose = {"services": {"a": {"image": "redis", "volumes": [volume]}}}
    error = _error(tmp_path, json.dumps(compose))
    assert error.code == "compose_unsupported"
    assert error.paths and error.paths[0].startswith("services.a.volumes[0]")


def test_mount_of_an_undeclared_volume_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    volumes: ['v:/data']\n")
    assert error.code == "compose_invalid"


@pytest.mark.parametrize(
    ("raw", "quantity"),
    [
        ("1gb", "1Gi"),
        ('"512m"', "512Mi"),
        ("64mb", "64Mi"),
        ("2g", "2Gi"),
        ("1024k", "1024Ki"),
        ("67108864", "67108864"),
    ],
)
def test_shm_size_becomes_a_kubernetes_quantity(tmp_path: Path, raw: str, quantity: str) -> None:
    spec = _spec(tmp_path, f"services:\n  a:\n    image: redis\n    shm_size: {raw}\n")
    assert _service(spec, "a")["shm_size"] == quantity


@pytest.mark.parametrize(
    ("limits", "task_toml"),
    [
        ("{cpus: '${CPUS}', memory: '${MEMORY}'}", "[environment]\ncpus = 1\nmemory_mb = 2048\n"),
        ("{cpus: '${CPUS:-4}', memory: '${MEMORY:-16G}'}", "[environment]\ncpus = 4\nmemory_mb = 16384\n"),
    ],
)
def test_limits_on_main_that_repeat_task_toml_are_accepted(tmp_path: Path, limits: str, task_toml: str) -> None:
    compose = f"services:\n  main:\n    deploy:\n      resources:\n        limits: {limits}\n  a:\n    image: redis\n"
    spec = _spec(tmp_path, compose, task_toml=task_toml)
    assert "resources" not in spec.get("main", {})


def test_limits_on_main_that_differ_from_task_toml_are_unsupported(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        "services:\n  main:\n    cpus: 2\n  a:\n    image: redis\n",
        task_toml="[environment]\ncpus = 1\n",
    )
    assert error.code == "compose_unsupported"
    assert error.paths == ("services.main: resource limits {'cpu': '2'} differ from task.toml's {'cpu': '1'}",)


@pytest.mark.parametrize(
    ("reservations", "path"),
    [
        (
            "{devices: [{driver: nvidia, count: all, capabilities: [gpu]}]}",
            "services.main.deploy.resources.reservations.devices",
        ),
        ("{cpus: '1'}", "services.main.deploy.resources.reservations"),
    ],
)
def test_reservations_on_main_are_unsupported(tmp_path: Path, reservations: str, path: str) -> None:
    compose = f"services:\n  main:\n    deploy:\n      resources:\n        reservations: {reservations}\n  a:\n    image: redis\n"
    error = _error(tmp_path, compose)
    assert error.code == "compose_unsupported"
    assert path in error.paths


def test_resources_from_cpus_mem_limit_and_deploy(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          a:
            image: redis
            cpus: 1.0
            mem_limit: 512m
          b:
            image: redis
            cpus: 0.5
            deploy:
              resources:
                limits: {cpus: "2", memory: 1g}
                reservations: {cpus: "0.25", memory: 256m}
        """,
    )

    assert _service(spec, "a")["resources"] == {"limits": {"cpu": "1", "memory": "512Mi"}}
    assert _service(spec, "b")["resources"] == {
        "limits": {"cpu": "2", "memory": "1Gi"},
        "requests": {"cpu": "0.25", "memory": "256Mi"},
    }


def test_main_settings(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            cap_add: [SYS_PTRACE]
            environment: {MODE: test}
            healthcheck:
              test: ["CMD", "test", "-f", "/tmp/ready"]
          a:
            image: redis
        """,
        dockerfile='FROM ubuntu\nENTRYPOINT ["/app/start.sh"]\nCMD ["serve"]\n',
    )

    assert spec["main"] == {
        "entrypoint": ["/app/start.sh"],
        "env": {"MODE": "test"},
        "add_capabilities": ["SYS_PTRACE"],
        # The default timeout (300 s) is left out of the kwarg.
        "readiness": {"exec": ["test", "-f", "/tmp/ready"]},
    }


def test_main_entrypoint_in_compose_wins_over_the_dockerfile(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        "services:\n  main:\n    entrypoint: ['/override']\n  a:\n    image: redis\n",
        dockerfile='FROM ubuntu\nENTRYPOINT ["/app/start.sh"]\n',
    )
    assert spec["main"]["entrypoint"] == ["/override"]


@pytest.mark.parametrize(
    ("dockerfile", "entrypoint"),
    [
        ('FROM a\nENTRYPOINT ["/x", "--y"]\n', ["/x", "--y"]),
        ("FROM a\nENTRYPOINT /x --y\n", ["sh", "-c", "/x --y"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT ["/two"]\n', ["/two"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT []\n', None),
        ('FROM a AS build\nENTRYPOINT ["/build"]\nFROM b\nRUN true\n', None),
        ('FROM a\nENTRYPOINT [\\\n  "/multi",\\\n  "line"\\\n]\n', ["/multi", "line"]),
        ('FROM a\n# ENTRYPOINT ["/comment"]\nRUN true\n', None),
        ('from a\nentrypoint ["/lower"]\n', ["/lower"]),
        ("FROM a\nENTRYPOINT [not json\n", ["sh", "-c", "[not json"]),
        ("FROM a\nRUN true\n", None),
    ],
)
def test_dockerfile_entrypoint(dockerfile: str, entrypoint: list[str] | None) -> None:
    assert dockerfile_entrypoint(dockerfile) == entrypoint


def test_ready_timeout_is_the_longest_healthcheck_budget_clamped(tmp_path: Path) -> None:
    def slow_service_spec(start: str, retries: int, main_check: bool, interval: str = "2s") -> ComposeServices:
        """The spec of a task whose ``slow`` service takes ``start + interval * retries`` to become healthy."""
        main = "healthcheck: {test: [CMD, true]}" if main_check else "environment: {A: b}"

        task = parse_compose(
            _task(
                tmp_path / f"{start}-{retries}-{main_check}",
                f"""
            services:
              main:
                {main}
              slow:
                image: redis
                healthcheck:
                  test: [CMD, redis-cli, ping]
                  start_period: {start}
                  interval: {interval}
                  retries: {retries}
            """,
            )
        )

        assert task is not None
        return to_compose_services(task)

    def main_timeout(start: str, retries: int) -> int:
        """The readiness timeout ``main`` gets when it has its own healthcheck."""
        readiness = slow_service_spec(start, retries, True).main.readiness
        assert readiness is not None
        return readiness.timeout_sec

    # 300 + 2 * 180 = 660, with main's own check to carry it.
    assert main_timeout("300s", 180) == 660

    # Without a main check, the services' wait is carried by services_timeout_sec.
    without_main_check = slow_service_spec("300s", 180, False)
    assert without_main_check.services_timeout_sec == 660
    assert without_main_check.main.readiness is None

    # A short wait leaves the default alone.
    assert slow_service_spec("5s", 30, False).services_timeout_sec is None
    assert main_timeout("5s", 30) == 300

    # An hour is the ceiling.
    assert main_timeout("1h", 1000) == 3600
    assert slow_service_spec("1m30s", 10, False, interval="1m").services_timeout_sec == 690


def test_run_once_healthchecks_do_not_count_toward_the_wait(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            depends_on:
              job: {condition: service_completed_successfully}
          job:
            image: acme/job:1
            healthcheck: {test: [CMD, true], start_period: 1h}
        """,
    )
    assert "services_timeout_sec" not in spec


def test_services_run_by_normalized_tag_unless_the_caller_supplies_an_image(tmp_path: Path) -> None:
    compose = "services:\n  a:\n    image: redis:7\n  b:\n    image: postgres:18\n"
    digest = "registry.example.com/mirror/redis@sha256:" + "c" * 64
    spec = _spec(tmp_path, compose, images={"a": digest})
    assert _service(spec, "a")["image"] == digest
    assert _service(spec, "b")["image"] == "docker.io/library/postgres:18"


def test_built_service_needs_its_image_from_the_caller(tmp_path: Path) -> None:
    compose = "services:\n  api:\n    build: ./api\n"
    task = _task(tmp_path, compose, files={"api/Dockerfile": "FROM scratch\n"})
    with pytest.raises(ComposeError) as caught:
        to_compose_services(parse_compose(task) or pytest.fail("not a compose task"))
    assert caught.value.code == "compose_build_not_prepared"
    assert "environment/api" in caught.value.detail
    spec = compose_services_kwarg(task, {"api": "registry.example.com/demo/api@sha256:" + "d" * 64})
    assert spec is not None
    assert _service(spec, "api")["image"].startswith("registry.example.com/demo/api@sha256:")


def test_provenance_records_each_service_and_its_image(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path, "services:\n  a:\n    image: redis:7\n  b:\n    build: ./b\n", files={"b/Dockerfile": "FROM x\n"}
    )
    pinned = "registry.example.com/demo/b@sha256:" + "e" * 64
    assert compose_provenance(to_compose_services(task, {"b": pinned})) == {
        "a": "docker.io/library/redis:7",
        "b": pinned,
    }


def test_build_sources(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          a: {build: ./api}
          b: {build: {context: ., dockerfile: odoo/Dockerfile, args: {V: 1}, target: prod, no_cache: true}}
          c: {image: acme/c:1, build: {context: ./c}}
        """,
    )

    assert task.services["a"].source == ComposeBuild(context_path="environment/api")
    assert task.services["b"].source == ComposeBuild(
        context_path="environment", dockerfile="odoo/Dockerfile", args={"V": "1"}, target="prod", no_cache=True
    )
    # Compose builds when both are given, and names the result by `image`.
    assert isinstance(task.services["c"].source, ComposeBuild)
    assert task.services["c"].source.context_path == "environment/c"


def test_platform_is_read_for_services_and_main(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main: {platform: linux/arm64}
          a: {image: redis, platform: linux/amd64}
          b: {image: redis}
        """,
    )

    assert task.main.platform == "linux/arm64"
    assert task.services["a"].platform == "linux/amd64"
    assert task.services["b"].platform is None


def test_a_platform_that_is_not_a_string_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    platform: [linux/amd64]\n")
    assert error.code == "compose_invalid"
    assert "services.a.platform" in error.detail


def test_published_image_source(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  a:\n    image: postgres:18\n")
    assert task.services["a"].source == ComposeImage(ref="docker.io/library/postgres:18")


# ---------------------------------------------------------------------------------------------------------------------
# Rejections


def test_every_unsupported_feature_is_listed(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            privileged: true
            cap_add: [SYS_PTRACE, NET_ADMIN]
            network_mode: host
            shm_size: 1gb
          a:
            image: redis
            privileged: true
            devices: ["/dev/fuse:/dev/fuse"]
            extends: {file: other.yaml, service: x}
            profiles: [debug]
            secrets: [s]
            configs: [c]
            cap_add: [SYS_PTRACE]
            network_mode: host
            user: root
            deploy: {replicas: 2}
          b:
            image: redis
            build: {context: ., args: {A: "${NOPE}"}, ssh: [default]}
        secrets: {s: {file: ./s}}
        networks: {backend: {ipam: {config: []}}}
        """,
    )

    assert error.code == "compose_unsupported"
    assert set(error.paths) >= {
        "services.main.privileged",
        "services.main.cap_add: NET_ADMIN",
        "services.main.network_mode",
        "services.main.shm_size",
        "services.a.privileged",
        "services.a.devices",
        "services.a.extends",
        "services.a.profiles",
        "services.a.secrets",
        "services.a.configs",
        "services.a.cap_add",
        "services.a.network_mode",
        "services.a.user",
        "services.a.deploy.replicas",
        "services.b.build.ssh",
        "secrets",
        "networks.backend.ipam",
    }


def test_features_with_no_effect_in_a_pod_are_accepted(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        x-common: &common
          restart: always
        services:
          main:
            network_mode: "service:a"
            pull_policy: build
            networks: [default]
            labels: {a: b}
            tty: true
            stdin_open: true
          a:
            <<: *common
            image: redis
            pull_policy: always
            container_name: cache
            hostname: cache
            labels: [a=b]
            logging: {driver: json-file}
            networks: {default: {}}
            network_mode: "service:b"
            healthcheck:
              test: [CMD, redis-cli, ping]
              timeout: 5s
              start_interval: 1s
          b:
            image: redis
            build: {context: ., no_cache: true}
        x-note: ignored
        """,
        images={"b": "registry.example.com/demo/b:1"},
    )

    assert [service["name"] for service in spec["services"]] == ["a", "b"]


@pytest.mark.parametrize(
    "build",
    ["https://github.com/acme/x.git", "git@github.com:acme/x.git", "..", "../outside", "/abs"],
)
def test_build_context_must_stay_under_environment(tmp_path: Path, build: str) -> None:
    error = _error(tmp_path, f"services:\n  a:\n    build: {build}\n")
    assert error.code == "compose_unsupported"


def test_dockerfile_must_stay_inside_its_context(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    build: {context: ., dockerfile: ../Dockerfile}\n")
    assert error.code == "compose_unsupported"


def test_service_with_neither_image_nor_build_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    environment: {A: b}\n")
    assert error.code == "compose_invalid"


def test_reserved_port_names_the_offending_key(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  odoo:\n    image: odoo:19\n    expose: ['8069', '18080']\n")
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_reserved_port"
    assert caught.value.paths == ("odoo.expose[1] = 18080",)


@pytest.mark.parametrize("port", ["44772", "15353", "18080"])
def test_every_pod_port_is_reserved(tmp_path: Path, port: str) -> None:
    task = _parsed(tmp_path, f"services:\n  a:\n    image: redis\n    ports: ['{port}']\n")
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_reserved_port"


def test_reserved_service_name(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  egress:\n    image: redis\n")
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_reserved_name"


def test_spec_rule_violations_are_invalid_and_name_the_service(tmp_path: Path) -> None:
    # Names are Kubernetes labels, so a compose name with an underscore can't be used.
    task = _parsed(tmp_path, "services:\n  my_db:\n    image: redis\n")
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_invalid"
    assert "services.my_db.name" in caught.value.detail


def test_two_services_on_one_port_are_invalid(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  a: {image: redis, expose: ['80']}\n  b: {image: redis, expose: ['80']}\n")
    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)
    assert caught.value.code == "compose_invalid"
    assert "same port" in caught.value.detail


def test_error_message_carries_the_code_and_compose_path(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    privileged: true\n")
    assert str(error) == (
        "compose_unsupported: these can't run as sandbox services (services.a.privileged)"
        " in demo/environment/docker-compose.yaml"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Hostile and malformed input


def test_yaml_aliases_that_expand_without_bound_are_refused(tmp_path: Path) -> None:
    # Each level holds nine references to the level below it, so nine levels expand to hundreds of millions of values.
    levels = ["l0: &l0 [x, x, x, x, x, x, x, x, x]"]
    levels += [f"l{level}: &l{level} [{', '.join([f'*l{level - 1}'] * 9)}]" for level in range(1, 9)]
    compose = "services:\n  db:\n    image: redis\nx-bomb:\n" + "".join(f"  {level}\n" for level in levels)

    error = _error(tmp_path, compose)

    assert error.code == "compose_invalid"
    assert "expands to more than" in error.detail


def test_a_file_nested_too_deeply_is_refused(tmp_path: Path) -> None:
    compose = "services: {db: {image: redis}}\nx-deep: " + "[" * 5000 + "]" * 5000 + "\n"

    assert _error(tmp_path, compose).code == "compose_invalid"


@pytest.mark.parametrize(
    "compose",
    [
        "services:\n  db:\n    image: redis\n    depends_on: {cache: service_healthy}\n  cache:\n    image: redis\n",
        "services:\n  db:\n    image: redis\n    deploy: {resources: [cpus]}\n",
        "services:\n  db:\n    image: redis\n    deploy: {resources: {limits: [cpus]}}\n",
        "services:\n  db:\n    image: redis\n    cpus: .nan\n",
        "services:\n  db:\n    image: redis\n    cpus: -1\n",
        "services:\n  main:\n    cap_add: SYS_PTRACE\n  db:\n    image: redis\n",
    ],
)
def test_malformed_values_are_a_typed_error_not_a_crash(tmp_path: Path, compose: str) -> None:
    assert _error(tmp_path, compose).code == "compose_invalid"


def test_networks_every_service_joins_are_accepted(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            networks: [internal]
          game:
            image: redis
            networks: {internal: {}}
        networks:
          internal: {driver: bridge, internal: false}
        """,
    )
    assert [service["name"] for service in spec["services"]] == ["game"]


def test_services_on_different_networks_are_unsupported(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            networks: [outside]
          bastion:
            image: redis
            networks: [outside, inside]
          db:
            image: redis
            networks: [inside]
        networks:
          outside: {}
          inside: {}
        """,
    )
    assert error.code == "compose_unsupported"
    assert set(error.paths) == {
        "services.bastion.networks: not the same networks as main",
        "services.db.networks: not the same networks as main",
    }


def test_fixed_addresses_and_network_options_are_unsupported(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            networks: {corp: {ipv4_address: 10.0.0.2}}
          a:
            image: redis
            networks: {corp: {aliases: [db]}}
        networks:
          corp:
            internal: true
            ipam: {config: [{subnet: 10.0.0.0/24}]}
        """,
    )
    assert set(error.paths) == {
        "services.main.networks.corp.ipv4_address",
        "services.a.networks.corp.aliases",
        "networks.corp.internal",
        "networks.corp.ipam",
    }


def test_a_network_that_is_not_declared_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  main:\n    networks: [nope]\n  a:\n    image: redis\n    networks: [nope]\n")
    assert error.code == "compose_invalid"
    assert "network 'nope' is not declared" in error.detail


def test_networks_that_are_not_names_are_unsupported_not_a_crash(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  db:\n    image: redis\n    networks: [{backend: {}}]\n")

    assert error.code == "compose_unsupported"
    assert "services.db.networks" in error.paths


@pytest.mark.parametrize(
    ("content", "complaint"),
    [
        (b"A=\xff\xfe\n", "can't read"),
        (b"A=1\n" * 100_000, "byte limit"),
    ],
    ids=["not-utf8", "oversized"],
)
def test_an_env_file_that_cannot_be_read_as_data_is_refused(tmp_path: Path, content: bytes, complaint: str) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n    env_file: data.env\n")
    (task / "environment" / "data.env").write_bytes(content)

    with pytest.raises(ComposeError) as caught:
        parse_compose(task)

    assert caught.value.code == "compose_invalid"
    assert complaint in caught.value.detail and "data.env" in caught.value.detail


def test_an_env_file_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n    env_file: data.env\n")
    target = tmp_path / "outside.env"
    target.write_text("A=1\n", encoding="utf-8")
    (task / "environment" / "data.env").symlink_to(target)

    with pytest.raises(ComposeError, match="data.env is a symlink"):
        parse_compose(task)


def test_dotenv_values_end_at_an_inline_comment_unless_quoted(tmp_path: Path) -> None:
    compose = "services:\n  db:\n    image: redis\n    environment: {A: '${A}', B: '${B}', C: '${C}'}\n"
    dotenv = 'A=plain # a note\nB="kept # as written"\nC=url#anchor\n'

    spec = _spec(tmp_path, compose, files={".env": dotenv})

    assert _service(spec, "db")["env"] == {"A": "plain", "B": "kept # as written", "C": "url#anchor"}


def test_a_service_waiting_on_an_after_main_service_is_after_main_too(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          loadgen:
            image: acme/loadgen:1
            depends_on:
              main: {condition: service_healthy}
          reporter:
            image: acme/reporter:1
            depends_on:
              loadgen: {condition: service_started}
        """,
    )

    assert [(service["name"], service["role"]) for service in spec["services"]] == [
        ("loadgen", "after_main"),
        ("reporter", "after_main"),
    ]


def test_main_waiting_on_a_service_that_waits_for_main_is_refused(tmp_path: Path) -> None:
    error = _error(
        tmp_path,
        """
        services:
          main:
            depends_on: [reporter]
          loadgen:
            image: acme/loadgen:1
            depends_on:
              main: {condition: service_healthy}
          reporter:
            image: acme/reporter:1
            depends_on: [loadgen]
        """,
    )

    assert error.code == "compose_invalid"
    assert error.paths == ("reporter",)


def test_a_dockerfile_instruction_continues_over_blank_and_comment_lines() -> None:
    dockerfile = 'ENTRYPOINT ["/start.sh", \\\n\n  # a note\n  "--flag"]\n'

    assert dockerfile_entrypoint(dockerfile) == ["/start.sh", "--flag"]


def test_a_quoted_dotenv_value_followed_by_a_comment_loses_its_quotes_and_the_comment(tmp_path: Path) -> None:
    compose = "services:\n  db:\n    image: pg:${TAG}\n    environment: {NOTE: '${NOTE}'}\n"
    dotenv = "TAG=\"18\" # pg\nNOTE='two words'   # c\n"

    spec = _spec(tmp_path, compose, files={".env": dotenv})

    assert _service(spec, "db")["image"] == "docker.io/library/pg:18"
    assert _service(spec, "db")["env"] == {"NOTE": "two words"}


def test_a_dotenv_value_quoted_across_lines_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n", files={".env": 'A="multi\nline"\n'})

    with pytest.raises(ComposeError, match="multi-line"):
        parse_compose(task)


def test_a_service_named_like_the_separate_verifiers_image_role_is_refused(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  verifier:\n    image: redis\n")

    with pytest.raises(ComposeError) as caught:
        to_compose_services(task)

    assert caught.value.code == "compose_reserved_name"
    assert caught.value.paths == ("verifier",)


@pytest.mark.parametrize("entrypoint", ["[]", "''"])
def test_an_empty_entrypoint_that_would_clear_the_images_is_unsupported(tmp_path: Path, entrypoint: str) -> None:
    error = _error(tmp_path, f"services:\n  db:\n    image: redis\n    entrypoint: {entrypoint}\n")

    assert error.code == "compose_unsupported"
    assert "services.db.entrypoint: empty" in error.paths[0]


def test_a_cpu_count_written_with_an_underscore_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  db:\n    image: redis\n    cpus: '1_0'\n")

    assert error.code == "compose_invalid"


# ---------------------------------------------------------------------------------------------------------------------
# End to end


def test_prebuilt_main_with_redis_and_postgres_sidecars(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          redis:
            image: redis:7
            healthcheck:
              test: ["CMD", "redis-cli", "ping"]
              interval: 5s
              retries: 3
          postgres:
            image: postgres:16
            environment:
              POSTGRES_PASSWORD: secret
            healthcheck:
              test: ["CMD-SHELL", "pg_isready -U postgres"]
              interval: 5s
              retries: 6
        """,
        task_toml='[environment]\ndocker_image = "registry.example.com/demo/main:1"\n',
        dockerfile=None,
    )

    assert [(service["name"], service["image"], service.get("role", "sidecar")) for service in spec["services"]] == [
        ("redis", "docker.io/library/redis:7", "sidecar"),
        ("postgres", "docker.io/library/postgres:16", "sidecar"),
    ]
    assert _service(spec, "redis")["readiness"] == {"exec": ["redis-cli", "ping"]}
    assert _service(spec, "postgres")["env"] == {"POSTGRES_PASSWORD": "secret"}
    assert "main" not in spec
    assert "services_timeout_sec" not in spec
    ComposeServices.model_validate(spec)


def test_built_main_with_a_sidecar_a_run_once_and_an_after_main_service(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            cap_add: [SYS_PTRACE]
            volumes: ["shared:/shared"]
            depends_on:
              api: {condition: service_healthy}
              seeder: {condition: service_completed_successfully}
          api:
            build: ./api
            healthcheck:
              test: ["CMD-SHELL", "curl -sf localhost:8000/health"]
          seeder:
            image: acme/seeder:1
            volumes: ["shared:/shared"]
            depends_on:
              api: {condition: service_healthy}
          loadgen:
            image: acme/loadgen:1
            depends_on:
              main: {condition: service_started}
        volumes:
          shared: {}
        """,
        images={"api": "registry.example.com/demo/api:1"},
        dockerfile='FROM python:3.12-slim\nENTRYPOINT ["/start.sh"]\n',
        files={"api/Dockerfile": DOCKERFILE},
    )

    assert [(service["name"], service.get("role", "sidecar")) for service in spec["services"]] == [
        ("api", "sidecar"),
        ("seeder", "run_once"),
        ("loadgen", "after_main"),
    ]
    assert _service(spec, "api")["image"] == "registry.example.com/demo/api:1"
    assert spec["volumes"] == ["shared"]
    assert spec["main"]["add_capabilities"] == ["SYS_PTRACE"]
    assert spec["main"]["volume_mounts"] == [{"name": "shared", "mount_path": "/shared"}]
    assert spec["main"]["entrypoint"] == ["/start.sh"]
    ComposeServices.model_validate(spec)


def test_compose_services_kwarg_round_trips_through_the_environment_model(tmp_path: Path) -> None:
    from nhx_sandbox.harbor_opensandbox_services import parse_compose_services

    spec = _spec(tmp_path, "services:\n  db:\n    image: postgres:16\n")
    assert parse_compose_services(spec).names() == ["db"]


@pytest.mark.skipif(not os.environ.get("TB3_TASKS_DIR"), reason="set TB3_TASKS_DIR to a Terminal-Bench tasks/ checkout")
def test_every_local_terminal_bench_compose_task_maps_to_a_valid_spec() -> None:
    """A local sweep: each task parses to a spec the environment accepts; only a reserved port may be rejected."""
    from nhx_sandbox.harbor_opensandbox_services import parse_compose_services

    tasks = sorted(
        path.parent.parent for path in Path(os.environ["TB3_TASKS_DIR"]).glob("*/environment/docker-compose.y*ml")
    )
    assert tasks, "no compose tasks under TB3_TASKS_DIR"
    rejected: dict[str, str] = {}
    for task_dir in tasks:
        parsed = parse_compose(task_dir)
        assert parsed is not None
        images = {name: f"registry.example.com/{task_dir.name}/{name}:test" for name in parsed.services}
        try:
            parse_compose_services(compose_services_kwarg(task_dir, images))
        except ComposeError as exc:
            rejected[task_dir.name] = exc.code
    assert set(rejected.values()) <= {"compose_reserved_port"}, rejected
