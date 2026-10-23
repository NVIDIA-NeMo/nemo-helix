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
          b: {build: {context: ., dockerfile: odoo/Dockerfile, args: {V: 1}, target: prod}}
          c: {image: acme/c:1, build: {context: ./c}}
          d: {image: postgres:18}
        """,
    )

    assert task.services["a"].source == ComposeBuild(context_path="environment/api")
    assert task.services["b"].source == ComposeBuild(
        context_path="environment", dockerfile="odoo/Dockerfile", args={"V": "1"}, target="prod"
    )
    # Compose builds when both are given, and names the result by `image`.
    assert task.services["c"].source == ComposeBuild(context_path="environment/c")
    assert task.services["d"].source == ComposeImage(reference="docker.io/library/postgres:18")


@pytest.mark.parametrize(
    ("dockerfile", "entrypoint"),
    [
        ('FROM a\nENTRYPOINT ["/x", "--y"]\n', ["/x", "--y"]),
        ("FROM a\nENTRYPOINT /x --y\n", ["sh", "-c", "/x --y"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT ["/two"]\n', ["/two"]),
        ('FROM a\nENTRYPOINT ["/one"]\nENTRYPOINT []\n', None),
        ('FROM a AS build\nENTRYPOINT ["/build"]\nFROM b\nRUN true\n', None),
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


def test_file_that_only_overrides_main_parses_but_has_no_services_to_run(tmp_path: Path) -> None:
    task = _parsed(tmp_path, "services:\n  main:\n    environment:\n      A: b\n")
    assert task.services == {} and task.main.env == {"A": "b"}
    with pytest.raises(ComposeError, match="compose_override_only"):
        to_compose_services(task)


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
        ({"interval": "5s"}, None),
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
    # `environment` wins over `env_file`, and a name with no value is left out, as on a host that doesn't define it.
    assert _service(spec, "b")["env"] == {"X": "from-compose", "Y": "1", "K": "v=w", "EMPTY": ""}
    assert spec["main"]["env"] == {"M": "1"}


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


def test_run_once_healthchecks_do_not_count_toward_the_wait(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            healthcheck: {test: [CMD, true]}
            depends_on:
              job: {condition: service_completed_successfully}
          job:
            image: acme/job:1
            healthcheck: {test: [CMD, true], start_period: 1h}
        """,
    )
    # The default timeout (300 s) is left out of the kwarg; an hour-long budget would have set one.
    assert spec["main"]["readiness"] == {"exec": ["true"]}


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
          restart: always
        services:
          main:
            network_mode: "service:a"
            pull_policy: build
            labels: {a: b}
          a:
            <<: *common
            image: redis
            pull_policy: always
            container_name: cache
            hostname: cache
            labels: [a=b]
            logging: {driver: json-file}
            network_mode: "service:b"
            healthcheck:
              test: [CMD, redis-cli, ping]
              timeout: 5s
              start_interval: 1s
          b:
            image: redis
            build: {context: .}
        x-note: ignored
        """,
        images={"b": "registry.example.com/demo/b:1"},
    )

    assert [service["name"] for service in spec["services"]] == ["a", "b"]


def test_a_service_may_be_named_verifier(tmp_path: Path) -> None:
    """kumo's game server is a service named `verifier`; the agent reaches it by that name."""
    spec = _spec(tmp_path, "services:\n  verifier:\n    image: redis\n")
    assert [service["name"] for service in spec["services"]] == ["verifier"]


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
        """,
    )

    assert error.code == "compose_unsupported"
    assert set(error.paths) >= {
        "services.main.privileged",
        "services.main.cap_add",
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
        "services.a.deploy",
        "services.b.build.ssh",
        "secrets",
    }


@pytest.mark.parametrize(
    ("compose", "code", "names"),
    [
        pytest.param(
            "services:\n  main:\n    command: [python, app.py]\n  db:\n    image: redis\n",
            "compose_unsupported",
            "services.main.command",
            id="mains-own-command",
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
        # Names are Kubernetes labels, so a compose name with an underscore can't be used.
        pytest.param(
            "services:\n  my_db:\n    image: redis\n", "compose_invalid", "services.my_db.name", id="spec-rule"
        ),
        pytest.param(
            "services:\n  a: {image: redis, expose: ['80']}\n  b: {image: redis, expose: ['80']}\n",
            "compose_invalid",
            "same port",
            id="two-services-on-one-port",
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
    ComposeServices.model_validate(spec)


def test_built_main_with_a_sidecar_a_run_once_and_an_after_main_service(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        """
        services:
          main:
            depends_on:
              api: {condition: service_healthy}
              seeder: {condition: service_completed_successfully}
          api:
            build: ./api
            healthcheck:
              test: ["CMD-SHELL", "curl -sf localhost:8000/health"]
          seeder:
            image: acme/seeder:1
            depends_on:
              api: {condition: service_healthy}
          loadgen:
            image: acme/loadgen:1
            depends_on:
              main: {condition: service_started}
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
    ComposeServices.model_validate(spec)


def test_compose_services_kwarg_round_trips_through_the_environment_model(tmp_path: Path) -> None:
    from nhx_sandbox.harbor_opensandbox_services import parse_compose_services

    spec = _spec(tmp_path, "services:\n  db:\n    image: postgres:16\n")
    assert parse_compose_services(spec).names() == ["db"]
