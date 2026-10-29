# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor task's compose file read service by service and mapped to ``ComposeServices``.

Every test writes its own task directory under ``tmp_path``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from nemo_evals.harbor.compose.images import dockerfile_entrypoint, normalize_image
from nemo_evals.harbor.compose.models import ComposeBuild, ComposeError, ComposeImage, ComposeTask
from nemo_evals.harbor.compose.parser import parse_compose
from nemo_evals.harbor.compose.spec import compose_provenance, compose_services_kwarg, to_compose_services
from nhx_sandbox.harbor_opensandbox_services import ComposeServices

DOCKERFILE = "FROM python:3.12-slim\n"
PREBUILT = '[environment]\ndocker_image = "registry.example.com/demo/main:1"\n'


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


def _parsed(tmp_path: Path, compose: str, **task_options: Any) -> ComposeTask:
    """The ``ComposeTask`` ``parse_compose`` reads from a task with this compose file."""
    task = parse_compose(_task(tmp_path, compose, **task_options))
    assert task is not None
    return task


def _spec(tmp_path: Path, compose: str, images: dict[str, str] | None = None, **task_options: Any) -> dict[str, Any]:
    """The ``compose_services`` kwarg for a compose file, as the dispatcher would pass it."""
    result = compose_services_kwarg(_task(tmp_path, compose, **task_options), images)
    assert result is not None
    return result


def _service(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """The service called ``name`` in a ``compose_services`` spec."""
    return next(service for service in spec["services"] if service["name"] == name)


def _error(tmp_path: Path, compose: str | None, **task_options: Any) -> ComposeError:
    """The error ``parse_compose`` raises for a task with this compose file."""
    with pytest.raises(ComposeError) as caught:
        parse_compose(_task(tmp_path, compose, **task_options))
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


@pytest.mark.parametrize(("no_cache", "expected"), [('"true"', True), ("yes", True), ('"Off"', False)])
def test_no_cache_is_read_as_compose_reads_a_boolean(tmp_path: Path, no_cache: str, expected: bool) -> None:
    task = _parsed(tmp_path, f"services:\n  a: {{build: {{context: ., no_cache: {no_cache}}}}}\n")
    assert task.services["a"].source == ComposeBuild(context_path="environment", no_cache=expected)


def test_invalid_image_name_is_a_typed_error() -> None:
    with pytest.raises(ComposeError) as caught:
        normalize_image("not a ref")
    assert caught.value.code == "compose_invalid"


def test_image_and_build_sources(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          a: {build: ./api}
          b: {build: {context: ., dockerfile: odoo/Dockerfile, args: {V: 1}, target: prod, no_cache: true}}
          c: {image: acme/c:1, build: {context: ./c}}
          d: {image: postgres:18}
        """,
    )

    assert task.services["a"].source == ComposeBuild(context_path="environment/api")
    assert task.services["b"].source == ComposeBuild(
        context_path="environment", dockerfile="odoo/Dockerfile", args={"V": "1"}, target="prod", no_cache=True
    )
    # Harbor runs `docker compose build` before starting a built task, so a build wins over `image`.
    assert task.services["c"].source == ComposeBuild(context_path="environment/c")
    assert task.services["d"].source == ComposeImage(reference="docker.io/library/postgres:18")


def test_a_prebuilt_task_runs_the_image_of_a_service_that_also_has_a_build(tmp_path: Path) -> None:
    # Harbor skips `docker compose build` for a prebuilt task, and `up` pulls the published image.
    task = _parsed(
        tmp_path, "services:\n  c: {image: acme/c:1, build: {context: ./c}}\n", task_toml=PREBUILT, dockerfile=None
    )
    assert task.services["c"].source == ComposeImage(reference="docker.io/acme/c:1")


@pytest.mark.parametrize(
    ("dockerfile", "entrypoint"),
    [
        ('FROM a\nENTRYPOINT ["/x", "--y"]\n', ["/x", "--y"]),
        ("FROM a\nENTRYPOINT /x --y\n", ["sh", "-c", "/x --y"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT ["/two"]\n', ["/two"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT []\n', None),
        ('FROM a AS build\nENTRYPOINT ["/build"]\nFROM b\nRUN true\n', None),
        # A stage built from an earlier one inherits its entrypoint; stage names match in any case.
        ('FROM a AS Base\nENTRYPOINT ["/base"]\nFROM --platform=linux/amd64 base\nRUN true\n', ["/base"]),
        ('FROM a AS base\nENTRYPOINT ["/base"]\nFROM base AS app\nENTRYPOINT ["/app"]\nFROM app\n', ["/app"]),
        ('FROM a\nENTRYPOINT [\\\n  "/multi",\\\n  "line"\\\n]\n', ["/multi", "line"]),
        ('ENTRYPOINT ["/start.sh", \\\n\n  # a note\n  "--flag"]\n', ["/start.sh", "--flag"]),
        ('FROM a\n# ENTRYPOINT ["/comment"]\nRUN true\n', None),
        ('from a\nentrypoint ["/lower"]\n', ["/lower"]),
        ("FROM a\nENTRYPOINT [not json\n", ["sh", "-c", "[not json"]),
        ("FROM a\nRUN true\n", None),
    ],
)
def test_dockerfile_entrypoint(dockerfile: str, entrypoint: list[str] | None) -> None:
    assert dockerfile_entrypoint(dockerfile) == entrypoint


# ---------------------------------------------------------------------------------------------------------------------
# main


def test_task_without_a_compose_file_is_single_container(tmp_path: Path) -> None:
    task = _task(tmp_path, None)
    assert parse_compose(task) is None
    assert compose_services_kwarg(task) is None


def test_a_file_that_only_changes_main_gives_a_spec_with_no_services(tmp_path: Path) -> None:
    # What swe-lancer's compose files do: start `main` with the task's own command, in its own directory.
    compose = (
        "services:\n  main:\n    command: /app/tests/run.sh\n    working_dir: /app\n    environment: [ISSUE_ID=1]\n"
    )
    services = to_compose_services(_parsed(tmp_path, compose))
    assert services.services == []
    assert services.main.command == ["/app/tests/run.sh"]
    assert (services.main.working_dir, services.main.env) == ("/app", {"ISSUE_ID": "1"})


def test_harbor_s_own_keep_alive_command_is_not_carried_as_main_s(tmp_path: Path) -> None:
    services = to_compose_services(_parsed(tmp_path, "services:\n  main:\n    environment: {A: b}\n"))
    assert services.main.command is None and services.main.env == {"A": "b"}


AGENT_DOCKERFILE = 'FROM scratch\nENTRYPOINT ["/agent"]\n'


@pytest.mark.parametrize(
    ("compose", "task_toml", "files", "expected"),
    [
        pytest.param(
            "services:\n  db:\n    image: redis\n",
            "",
            {},
            ("Dockerfile", ComposeBuild(context_path="environment"), None, None),
            id="built-from-environment-by-default",
        ),
        pytest.param(
            "services:\n  main:\n    build: {dockerfile: Dockerfile.agent}\n  db:\n    image: redis\n",
            "",
            {"Dockerfile.agent": AGENT_DOCKERFILE},
            (
                "Dockerfile.agent",
                ComposeBuild(context_path="environment", dockerfile="Dockerfile.agent"),
                None,
                ["/agent"],
            ),
            id="task-chooses-the-dockerfile",
        ),
        pytest.param(
            "services:\n  main:\n    build: {context: ./agent, dockerfile: docker/Main.Dockerfile}\n"
            "  db:\n    image: redis\n",
            "",
            {"agent/docker/Main.Dockerfile": DOCKERFILE},
            (
                "agent/docker/Main.Dockerfile",
                ComposeBuild(context_path="environment/agent", dockerfile="docker/Main.Dockerfile"),
                None,
                None,
            ),
            id="context-and-dockerfile-combine",
        ),
        pytest.param(
            "services:\n  main:\n    build: {dockerfile: Dockerfile.agent}\n  db:\n    image: redis\n",
            PREBUILT,
            {},
            (None, None, "registry.example.com/demo/main:1", None),
            id="prebuilt-ignores-mains-build",
        ),
        pytest.param(
            "services:\n  main:\n    image: ghcr.io/org/sort:2\n  db:\n    image: redis\n",
            PREBUILT,
            {},
            (None, None, "ghcr.io/org/sort:2", None),
            id="task-replaces-the-prebuilt-image",
        ),
    ],
)
def test_mains_build_image_and_entrypoint(
    tmp_path: Path, compose: str, task_toml: str, files: dict[str, str], expected: tuple
) -> None:
    """What builds ``main`` and what it runs: (Dockerfile, build, prebuilt image, entrypoint)."""
    task = _parsed(tmp_path, compose, task_toml=task_toml, files=files, dockerfile=None if task_toml else DOCKERFILE)
    assert (task.main_dockerfile, task.main_build, task.main_image, task.main.entrypoint) == expected


def test_main_dockerfile_is_needed_unless_only_images_are_planned(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n", dockerfile=None)
    with pytest.raises(ComposeError, match="compose_invalid: .*main's Dockerfile Dockerfile is not in the task"):
        parse_compose(task)

    planned = parse_compose(task, read_entrypoint=False)
    assert planned is not None
    assert planned.main_build == ComposeBuild(context_path="environment") and planned.main.entrypoint is None


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


@pytest.mark.parametrize(
    ("dockerfile", "why"),
    [
        pytest.param("FROM a\n" + "#" * (256 * 1024), "over the 262144 byte limit", id="oversized"),
        pytest.param(None, "is a symlink", id="symlinked"),
    ],
)
def test_mains_dockerfile_is_read_like_the_tasks_other_files(tmp_path: Path, dockerfile: str | None, why: str) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n", dockerfile=dockerfile, files={"real": "FROM a\n"})
    if dockerfile is None:
        (task / "environment" / "Dockerfile").symlink_to("real")

    with pytest.raises(ComposeError) as caught:
        parse_compose(task)
    assert caught.value.code == "compose_invalid" and why in str(caught.value)


def test_main_entrypoint_in_compose_wins_over_the_dockerfile(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        "services:\n  main:\n    entrypoint: ['/override']\n  a:\n    image: redis\n",
        dockerfile='FROM ubuntu\nENTRYPOINT ["/app/start.sh"]\n',
    )
    assert spec["main"]["entrypoint"] == ["/override"]


# ---------------------------------------------------------------------------------------------------------------------
# Services


def test_services_get_roles_and_a_start_order(tmp_path: Path) -> None:
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
          migrate:
            image: acme/migrate:1
          app:
            image: acme/app:1
            depends_on:
              migrate: {condition: service_completed_successfully}
          reporter:
            image: acme/reporter:1
            depends_on: [loadgen]
        """,
    )

    # Each after its dependencies. `run_once`: something waits for it to finish, whether main (seeder) or another
    # service (migrate). `after_main`: it waits for main, or for a service that does (reporter).
    assert [(service["name"], service.get("role", "sidecar")) for service in spec["services"]] == [
        ("kafka", "sidecar"),
        ("seeder", "run_once"),
        ("loadgen", "after_main"),
        ("other", "sidecar"),
        ("migrate", "run_once"),
        ("app", "sidecar"),
        ("reporter", "after_main"),
    ]
    assert "readiness" not in _service(spec, "seeder")
    assert _service(spec, "kafka")["readiness"] == {"exec": ["sh", "-c", "nc -z localhost 9092"]}


def test_depends_on_list_form_means_started(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  main:\n    depends_on: [db]\n  db:\n    image: redis\n")
    assert task.main.depends_on == {"db": "service_started"}


@pytest.mark.parametrize(
    ("healthcheck", "readiness"),
    [
        ({"test": ["CMD", "curl", "-f", "http://localhost"]}, {"exec": ["curl", "-f", "http://localhost"]}),
        ({"test": ["CMD-SHELL", "pg_isready -U x"]}, {"exec": ["sh", "-c", "pg_isready -U x"]}),
        ({"test": "pg_isready -U x"}, {"exec": ["sh", "-c", "pg_isready -U x"]}),
        ({"test": ["NONE"]}, None),
        ({"disable": True}, None),
    ],
)
def test_healthcheck_becomes_readiness(tmp_path: Path, healthcheck: dict, readiness: dict | None) -> None:
    spec = _spec(tmp_path, json.dumps({"services": {"db": {"image": "redis", "healthcheck": healthcheck}}}))
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


def test_environment_forms_scalars_and_env_files(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            env_file: main.env
          a:
            image: redis
            environment:
              PORT: 8080
              DEBUG: true
              RATIO: 0.5
              NAME: "x y"
          b:
            image: redis
            env_file: [b.env]
            environment:
              - K=v=w
              - EMPTY=
              - FROM_HOST
              - X=from-compose
        """,
        files={"b.env": "X=from-file\nY=1\n", "main.env": "M=1\n"},
    )

    assert _service(spec, "a")["env"] == {"PORT": "8080", "DEBUG": "true", "RATIO": "0.5", "NAME": "x y"}
    # `environment` wins over `env_file`, and a name with no value is left out when Compose has no value for it.
    assert _service(spec, "b")["env"] == {"X": "from-compose", "Y": "1", "K": "v=w", "EMPTY": ""}
    assert spec["main"]["env"] == {"M": "1"}


def test_a_value_less_entry_takes_compose_s_value_or_unsets_the_env_file_one(tmp_path: Path) -> None:
    # `docker compose up` gives the container TOKEN from `.env`, and no A even though `a.env` sets it.
    task = _parsed(
        tmp_path,
        """
        services:
          db:
            image: redis
            env_file: a.env
            environment:
              TOKEN:
              A: null
            build: {context: ., args: [TOKEN, UNSET]}
        """,
        files={".env": "TOKEN=from-dotenv\n", "a.env": "A=from-file\n"},
    )

    assert task.services["db"].env == {"TOKEN": "from-dotenv"}
    assert isinstance(task.services["db"].source, ComposeBuild)
    assert task.services["db"].source.args == {"TOKEN": "from-dotenv"}


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


def test_ready_timeout_is_the_longest_healthcheck_budget_clamped(tmp_path: Path) -> None:
    def main_timeout(start: str, retries: int) -> int:
        """``main``'s readiness timeout when a ``slow`` service takes ``start + 2s * retries`` to become healthy."""
        compose = f"""
            services:
              main:
                healthcheck: {{test: [CMD, true]}}
              slow:
                image: redis
                healthcheck: {{test: [CMD, redis-cli, ping], start_period: {start}, interval: 2s, retries: {retries}}}
            """
        readiness = to_compose_services(_parsed(tmp_path / f"{start}-{retries}", compose)).main.readiness
        assert readiness is not None
        return readiness.timeout_sec

    assert main_timeout("300s", 180) == 660  # 300 + 2 * 180
    assert main_timeout("5s", 30) == 300  # a short wait leaves the default alone
    assert main_timeout("1h", 1000) == 3600  # an hour is the ceiling


def test_without_a_main_healthcheck_the_wait_is_carried_by_services_timeout_sec(tmp_path: Path) -> None:
    def services_timeout(start: str, retries: int) -> int | None:
        """``services_timeout_sec`` when a ``slow`` service takes ``start + 2s * retries`` and ``main`` has no check."""
        compose = f"""
            services:
              slow:
                image: redis
                healthcheck: {{test: [CMD, redis-cli, ping], start_period: {start}, interval: 2s, retries: {retries}}}
            """
        services = to_compose_services(_parsed(tmp_path / f"{start}-{retries}", compose))
        assert services.main.readiness is None
        return services.services_timeout_sec

    assert services_timeout("300s", 180) == 660  # 300 + 2 * 180
    assert services_timeout("5s", 30) is None  # within the sandbox's default wait
    assert services_timeout("1h", 1000) == 3600  # an hour is the ceiling


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
    # An hour-long budget would have set a timeout; the default (300 s) is left out of the kwarg.
    assert "services_timeout_sec" not in spec


def test_services_run_by_normalized_tag_unless_the_caller_supplies_an_image(tmp_path: Path) -> None:
    compose = "services:\n  a:\n    image: redis:7\n  b:\n    image: postgres:18\n"
    digest = "registry.example.com/mirror/redis@sha256:" + "c" * 64
    spec = _spec(tmp_path, compose, images={"a": digest})
    assert _service(spec, "a")["image"] == digest
    assert _service(spec, "b")["image"] == "docker.io/library/postgres:18"


def test_built_service_needs_its_image_from_the_caller(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  api:\n    build: ./api\n", files={"api/Dockerfile": "FROM scratch\n"})
    with pytest.raises(ComposeError) as caught:
        to_compose_services(parse_compose(task) or pytest.fail("not a compose task"))
    assert caught.value.code == "compose_build_not_prepared"
    assert "environment/api" in caught.value.detail

    pinned = "registry.example.com/demo/api@sha256:" + "d" * 64
    assert compose_provenance(to_compose_services(parse_compose(task) or pytest.fail(), {"api": pinned})) == {
        "api": pinned
    }


def test_features_with_no_effect_in_a_pod_are_accepted(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        x-common: &common
          logging: {driver: json-file}
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


def test_dollar_signs_reach_the_pod_as_compose_passes_them(tmp_path: Path) -> None:
    # Kubernetes would expand `$(TOKEN)` and turn `$$` into `$`; Compose does neither, so each `$` is doubled.
    spec = _spec(
        tmp_path,
        """
        services:
          db:
            image: redis
            entrypoint: ["/entry.sh", "$$(TOKEN)"]
            command: [echo, "$$(TOKEN)", "cost: $$$$5"]
            environment: {TOKEN: secret, ECHO: "$$(TOKEN)"}
            healthcheck:
              test: [CMD-SHELL, "test $$(cat /ready) = ok"]
        """,
    )

    db = _service(spec, "db")
    assert db["command"] == ["/entry.sh", "$$(TOKEN)"]
    assert db["args"] == ["echo", "$$(TOKEN)", "cost: $$$$5"]
    assert db["env"] == {"TOKEN": "secret", "ECHO": "$$(TOKEN)"}
    assert db["readiness"] == {"exec": ["sh", "-c", "test $$(cat /ready) = ok"]}


@pytest.mark.parametrize(("count", "accepted"), [(16, True), (17, False)])
def test_a_file_with_more_services_than_a_sandbox_runs_is_refused_first(
    tmp_path: Path, count: int, accepted: bool
) -> None:
    # The services form one reverse `depends_on` chain, the case whose start order is slowest to work out.
    compose = "services:\n" + "".join(
        f"  s{index}:\n    image: redis\n" + (f"    depends_on: [s{index + 1}]\n" if index < count - 1 else "")
        for index in range(count)
    )
    if accepted:
        assert len(_spec(tmp_path, compose)["services"]) == count
    else:
        error = _error(tmp_path, compose)
        assert error.code == "compose_unsupported" and "17 services besides main" in error.detail


def test_a_service_may_be_named_verifier(tmp_path: Path) -> None:
    """kumo's game server is a service named `verifier`; the agent reaches it by that name."""
    spec = _spec(tmp_path, "services:\n  verifier:\n    image: redis\n")
    assert [service["name"] for service in spec["services"]] == ["verifier"]


# ---------------------------------------------------------------------------------------------------------------------
# Volumes, resources, networks and Harbor's conventions


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


def test_a_long_form_mount_reads_read_only_and_nocopy_as_compose_does(tmp_path: Path) -> None:
    # `nocopy` asks for the volume to start empty, which a pod's volume always does.
    spec = _spec(
        tmp_path,
        """
        services:
          a:
            image: redis
            volumes:
              - {type: volume, source: data, target: /ro, read_only: "yes"}
              - {type: volume, source: data, target: /rw, read_only: "false", volume: {nocopy: true}}
        volumes:
          data: {}
        """,
    )
    assert _service(spec, "a")["volume_mounts"] == [
        {"name": "data", "mount_path": "/ro", "read_only": True},
        {"name": "data", "mount_path": "/rw"},
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
    ("shm_size", "quantity"),
    [
        ("1gb", "1Gi"),
        ('"512m"', "512Mi"),
        ("64mb", "64Mi"),
        ("2g", "2Gi"),
        ("1024k", "1024Ki"),
        ("67108864", "67108864"),
    ],
)
def test_shm_size_becomes_a_kubernetes_quantity(tmp_path: Path, shm_size: str, quantity: str) -> None:
    spec = _spec(tmp_path, f"services:\n  a:\n    image: redis\n    shm_size: {shm_size}\n")
    assert _service(spec, "a")["shm_size"] == quantity


def test_shm_size_with_a_volume_at_dev_shm_is_unsupported(tmp_path: Path) -> None:
    # The sandbox mounts /dev/shm itself for `shm_size`, and a container can't have two mounts at one path.
    compose = "services:\n  a:\n    image: redis\n    {shm_size}volumes: [data:/dev/shm]\nvolumes:\n  data: {{}}\n"
    with pytest.raises(ComposeError) as caught:
        to_compose_services(_parsed(tmp_path / "both", compose.format(shm_size="shm_size: 1g\n    ")))
    assert caught.value.code == "compose_unsupported"
    assert "'/dev/shm' is used by more than one volume mount" in caught.value.detail

    spec = _spec(tmp_path / "volume-only", compose.format(shm_size=""))
    assert _service(spec, "a")["volume_mounts"] == [{"name": "data", "mount_path": "/dev/shm"}]


@pytest.mark.parametrize("size", ["mem_limit: .nan", "mem_limit: 2.0", "shm_size: .inf"])
def test_a_float_size_sets_no_limit_as_in_compose(tmp_path: Path, size: str) -> None:
    spec = _spec(tmp_path, f"services:\n  a:\n    image: redis\n    {size}\n")
    assert not {"resources", "shm_size"} & _service(spec, "a").keys()


@pytest.mark.parametrize(
    ("limits", "task_toml"),
    [
        ("{cpus: '${CPUS}', memory: '${MEMORY}'}", "[environment]\ncpus = 1\nmemory_mb = 2048\n"),
        ("{cpus: '${CPUS:-4}', memory: '${MEMORY:-16G}'}", "[environment]\ncpus = 4\nmemory_mb = 16384\n"),
        # The same limits spelled differently.
        ("{cpus: '2.0', memory: 2g}", "[environment]\ncpus = 2\nmemory_mb = 2048\n"),
        ("{memory: '2147483648'}", "[environment]\nmemory_mb = 2048\n"),
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
            cpus: 2
            mem_limit: 1024m
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


def test_platform_is_read_for_services_and_main(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path,
        """
        services:
          main: {platform: linux/arm64}
          a: {image: redis, platform: linux/amd64}
          b: {image: redis}
          c: {image: redis, platform: "${UNSET_PLATFORM}"}
        """,
    )

    assert task.main.platform == "linux/arm64"
    assert task.services["a"].platform == "linux/amd64"
    assert task.services["b"].platform is None
    # An unset variable leaves the platform empty, which names none.
    assert task.services["c"].platform is None


def test_a_platform_that_is_not_a_string_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    platform: [linux/amd64]\n")
    assert error.code == "compose_invalid"
    assert "services.a.platform" in error.detail


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


def test_main_in_another_service_s_network_namespace_has_that_service_s_networks(tmp_path: Path) -> None:
    compose = """
        services:
          main:
            network_mode: "service:proxy"
          proxy:
            image: redis
            network_mode: "service:db"
          db:
            image: redis
            networks: [backend]
          cache:
            image: redis
            networks: [{network}]
        networks:
          backend: {{}}
          other: {{}}
        """
    spec = _spec(tmp_path / "same", compose.format(network="backend"))
    assert {service["name"] for service in spec["services"]} == {"proxy", "db", "cache"}

    error = _error(tmp_path / "other", compose.format(network="other"))
    assert error.paths == ("services.cache.networks: not the same networks as main",)


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


def test_a_cpu_count_written_with_an_underscore_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  db:\n    image: redis\n    cpus: '1_0'\n")

    assert error.code == "compose_invalid"


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


@pytest.mark.parametrize(
    ("compose", "code", "names"),
    [
        pytest.param(
            "services:\n  main:\n    command: []\n",
            "compose_unsupported",
            "services.main.command: an empty command",
            id="mains-empty-command",
        ),
        pytest.param(
            "services:\n  main:\n    command: null\n",
            "compose_unsupported",
            "services.main.command: null",
            id="mains-null-command",
        ),
        pytest.param(
            "services:\n  main:\n    working_dir: app\n",
            "compose_unsupported",
            "services.main.working_dir: a relative working directory",
            id="mains-relative-working-dir",
        ),
        pytest.param(
            "services:\n  a: {image: x, depends_on: [b]}\n  b: {image: x, depends_on: [a]}\n",
            "compose_invalid",
            "cycle",
            id="dependency-cycle",
        ),
        pytest.param(
            "services:\n  a:\n    image: redis\n    depends_on: [ghost]\n",
            "compose_invalid",
            "'ghost' is not a service",
            id="unknown-dependency",
        ),
        pytest.param(
            "services:\n  main:\n    depends_on: [a]\n  a:\n    image: redis\n    depends_on: [main]\n",
            "compose_invalid",
            "depend on each other",
            id="service-and-main-wait-on-each-other",
        ),
        pytest.param(
            "services:\n  main:\n    depends_on: [reporter]\n  loadgen:\n    image: x\n    depends_on: [main]\n"
            "  reporter:\n    image: x\n    depends_on: [loadgen]\n",
            "compose_invalid",
            "(reporter)",
            id="main-waits-on-a-service-that-waits-for-main",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    depends_on: {cache: service_healthy}\n  cache:\n    image: redis\n",
            "compose_invalid",
            "services.db.depends_on.cache",
            id="malformed-depends-on",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    deploy: {resources: [cpus]}\n",
            "compose_invalid",
            "services.db.deploy.resources",
            id="malformed-resources",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    deploy: {resources: {limits: [cpus]}}\n",
            "compose_invalid",
            "services.db.deploy.resources.limits",
            id="malformed-limits",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    cpus: .nan\n",
            "compose_invalid",
            "services.db.cpus",
            id="cpus-not-a-number",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    cpus: -1\n",
            "compose_invalid",
            "services.db.cpus",
            id="cpus-negative",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    cpus: 1" + "0" * 400 + "\n",
            "compose_invalid",
            "services.db.cpus: not a number",
            id="cpus-too-large-for-a-float",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    cpus: 0.5\n    deploy: {resources: {limits: {cpus: '2'}}}\n",
            "compose_invalid",
            "services.db.cpus: can't set distinct values on 'cpus' and 'deploy.resources.limits.cpus'",
            id="cpus-twice",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    mem_limit: 512m\n    deploy: {resources: {limits: {memory: 1g}}}\n",
            "compose_invalid",
            "services.db.mem_limit: can't set distinct values on 'mem_limit' and 'deploy.resources.limits.memory'",
            id="memory-twice",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    deploy: {resources: {limits: {memory: .inf}}}\n",
            "compose_invalid",
            "services.db.deploy.resources.limits.memory: must be a string",
            id="deploy-memory-a-number",
        ),
        pytest.param(
            "services:\n  a:\n    image: redis\n    volumes: [{type: volume, source: d, target: /d, volume: {subpath: x}}]\n"
            "volumes:\n  d: {}\n",
            "compose_unsupported",
            "services.a.volumes[0].volume.subpath",
            id="volume-subpath",
        ),
        pytest.param(
            "services:\n  a:\n    image: redis\n    volumes: [{type: volume, source: d, target: /d, read_only: t}]\n"
            "volumes:\n  d: {}\n",
            "compose_invalid",
            "services.a.volumes[0].read_only: not a boolean",
            id="read-only-not-a-boolean",
        ),
        pytest.param(
            "services:\n  a:\n    build: {context: ., no_cache: 1}\n",
            "compose_invalid",
            "services.a.build.no_cache: not a boolean",
            id="no-cache-not-a-boolean",
        ),
        pytest.param(
            "services:\n  a:\n    image: redis\n    network_mode: service:missing\n",
            "compose_invalid",
            "services.a.network_mode: depends on undefined service 'missing'",
            id="network-mode-undefined-service",
        ),
        pytest.param(
            "services:\n  a:\n    image: redis\n    network_mode: service:b\n"
            "  b:\n    image: redis\n    network_mode: service:a\n",
            "compose_invalid",
            "services.a.network_mode: dependency cycle detected: a -> b -> a",
            id="network-mode-cycle",
        ),
        pytest.param(
            "services:\n  main:\n    cap_add: SYS_PTRACE\n  db:\n    image: redis\n",
            "compose_invalid",
            "services.main.cap_add",
            id="cap-add-not-a-list",
        ),
        pytest.param(
            'services:\n  a:\n    image: redis\n    ports: ["8000-8005:8000-8005"]\n',
            "compose_unsupported",
            "port ranges",
            id="port-range",
        ),
        pytest.param(
            "services:\n  a:\n    build: {context: ., dockerfile: ../Dockerfile}\n",
            "compose_unsupported",
            "outside its context",
            id="dockerfile-outside-its-context",
        ),
        pytest.param(
            "services:\n  a:\n    environment: {A: b}\n",
            "compose_invalid",
            "neither `image` nor `build`",
            id="no-image-or-build",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    entrypoint: []\n",
            "compose_unsupported",
            "services.db.entrypoint: empty",
            id="empty-entrypoint-list",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    entrypoint: ''\n",
            "compose_unsupported",
            "services.db.entrypoint: empty",
            id="empty-entrypoint-string",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    restart: always\n",
            "compose_unsupported",
            "services.db.restart",
            id="restart",
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    healthcheck: {interval: 1s}\n",
            "compose_unsupported",
            "services.db.healthcheck: no `test`",
            id="healthcheck-without-test",
        ),
        pytest.param(
            "services:\n  main:\n    depends_on: {db: {condition: service_healthy}}\n  db:\n    image: redis\n",
            "compose_unsupported",
            "services.main.depends_on.db: db has no healthcheck test",
            id="healthy-wait-on-a-service-without-a-healthcheck",
        ),
        # `checker` would start next to `seed`, without waiting for it to finish.
        pytest.param(
            """
            services:
              seed:
                image: x
                depends_on: {main: {condition: service_healthy}}
              checker:
                image: x
                depends_on: {seed: {condition: service_completed_successfully}}
            """,
            "compose_unsupported",
            "services.checker.depends_on.seed: service_completed_successfully between two services that start after main",
            id="after-main-waits-for-another-to-finish",
        ),
        pytest.param(
            """
            services:
              worker:
                image: x
                depends_on: [main]
                healthcheck: {test: [CMD, "true"]}
              reporter:
                image: x
                depends_on: {worker: {condition: service_healthy}}
            """,
            "compose_unsupported",
            "services.reporter.depends_on.worker: service_healthy between two services",
            id="after-main-waits-for-another-to-be-healthy",
        ),
        *(
            pytest.param(
                f"services:\n  db:\n    image: redis\n    healthcheck: {{test: [CMD, 'true'], interval: {value}}}\n",
                "compose_invalid",
                "services.db.healthcheck.interval: not a duration",
                id=f"numeric-duration-{value}",
            )
            for value in ("5", ".inf", ".nan")
        ),
        pytest.param(
            "services:\n  db:\n    image: redis\n    expose: ['\u00b2']\n",
            "compose_unsupported",
            "services.db.expose[0]: port ranges and names",
            id="non-ascii-digit-port",
        ),
    ],
)
def test_a_file_the_parser_refuses_names_why(tmp_path: Path, compose: str, code: str, names: str) -> None:
    error = _error(tmp_path, compose)
    assert error.code == code
    assert names in str(error)


@pytest.mark.parametrize(
    "build", ["https://github.com/acme/x.git", "git@github.com:acme/x.git", "..", "../outside", "/abs"]
)
def test_build_context_must_stay_under_environment(tmp_path: Path, build: str) -> None:
    assert _error(tmp_path, f"services:\n  a:\n    build: {build}\n").code == "compose_unsupported"


@pytest.mark.parametrize(
    ("compose", "code", "names"),
    [
        pytest.param(
            "services:\n  odoo:\n    image: odoo:19\n    expose: ['8069', '18080']\n",
            "compose_reserved_port",
            "odoo.expose[1] = 18080",
            id="reserved-port-names-the-key",
        ),
        *(
            pytest.param(
                f"services:\n  a:\n    image: redis\n    ports: ['{port}']\n",
                "compose_reserved_port",
                port,
                id=f"pod-port-{port}",
            )
            for port in ("44772", "15353", "18080")
        ),
        pytest.param("services:\n  egress:\n    image: redis\n", "compose_reserved_name", "egress", id="reserved-name"),
        # Names are Kubernetes labels, so a compose name with an underscore can't be used. Compose accepts both files.
        pytest.param(
            "services:\n  my_db:\n    image: redis\n", "compose_unsupported", "services.my_db.name", id="spec-rule"
        ),
        pytest.param(
            "services:\n  a: {image: redis, expose: ['80']}\n  b: {image: redis, expose: ['80']}\n",
            "compose_unsupported",
            "same port",
            id="two-services-on-one-port",
        ),
        pytest.param(
            "services:\n  main:\n    expose: ['8080']\n  api:\n    image: redis\n    ports: ['9000:8080']\n",
            "compose_unsupported",
            "main.expose[0] = 8080",
            id="main-and-a-service-on-one-port",
        ),
        pytest.param(
            "services:\n  main:\n    ports: ['44772']\n  db:\n    image: redis\n",
            "compose_reserved_port",
            "main.ports[0] = 44772",
            id="main-on-a-pod-port",
        ),
    ],
)
def test_a_task_the_sandbox_spec_refuses_names_why(tmp_path: Path, compose: str, code: str, names: str) -> None:
    with pytest.raises(ComposeError) as caught:
        to_compose_services(_parsed(tmp_path, compose))
    assert caught.value.code == code
    assert names in str(caught.value)


def test_error_message_carries_the_code_and_compose_path(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: redis\n    privileged: true\n")
    assert str(error) == (
        "compose_unsupported: these can't run as sandbox services (services.a.privileged)"
        " in demo/environment/docker-compose.yaml"
    )


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
        task_toml=PREBUILT,
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
    assert spec["main"]["entrypoint"] == ["/start.sh"]
    assert spec["volumes"] == ["shared"]
    assert spec["main"]["add_capabilities"] == ["SYS_PTRACE"]
    assert spec["main"]["volume_mounts"] == [{"name": "shared", "mount_path": "/shared"}]
    ComposeServices.model_validate(spec)


def test_compose_services_kwarg_round_trips_through_the_environment_model(tmp_path: Path) -> None:
    from nhx_sandbox.harbor_opensandbox_services import parse_compose_services

    spec = _spec(tmp_path, "services:\n  db:\n    image: postgres:16\n")
    assert parse_compose_services(spec).names() == ["db"]


def test_main_and_a_service_may_use_different_ports(tmp_path: Path) -> None:
    task = _parsed(
        tmp_path, "services:\n  main:\n    expose: ['8080']\n  api:\n    image: redis\n    expose: ['9000']\n"
    )

    assert [port.port for port in task.main.ports] == [8080]
    to_compose_services(task)
