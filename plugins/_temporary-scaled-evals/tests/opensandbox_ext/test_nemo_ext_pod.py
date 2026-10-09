# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox server extension's pod builder and pod state reader."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("kubernetes")
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

from kubernetes.client import (
    V1ContainerState,
    V1ContainerStateRunning,
    V1ContainerStateTerminated,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1Pod,
    V1PodStatus,
)
from nemo_ext_pod import add_services, service_states
from scaled_evals.harbor_opensandbox_services import ComposeServices, parse_compose_services

SERVICES = parse_compose_services(
    {
        "volumes": ["shared"],
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "pw"},
                "ports": [5432],
                "readiness": {"exec": ["pg_isready"]},
                "volume_mounts": [{"name": "shared", "mount_path": "/data"}],
                "shm_size": "1Gi",
            },
            {"name": "migrate", "image": "docker.io/library/busybox:1", "command": ["true"], "role": "run_once"},
            {"name": "cache", "image": "docker.io/library/redis:7"},
            {
                "name": "worker",
                "image": "docker.io/library/redis:7",
                "role": "after_main",
                "readiness": {"exec": ["redis-cli", "ping"]},
                "resources": {"limits": {"cpu": "1"}},
            },
        ],
        "main": {"volume_mounts": [{"name": "shared", "mount_path": "/shared"}], "add_capabilities": ["SYS_PTRACE"]},
    }
)


def stock_pod_spec(*, egress: bool = True) -> dict[str, Any]:
    """The parts of the pod spec opensandbox-server 0.2.1 builds that ``add_services`` touches."""
    containers: list[dict[str, Any]] = [
        {
            "name": "sandbox",
            "image": "registry.example.com/task:1",
            "securityContext": {"capabilities": {"drop": ["NET_ADMIN"]}},
        }
    ]
    if egress:
        containers.append(
            {
                "name": "egress",
                "image": "registry.example.com/egress:1",
                "readinessProbe": {"httpGet": {"path": "/healthz", "port": 18080}, "periodSeconds": 1},
            }
        )
    return {
        "initContainers": [{"name": "execd-installer", "image": "registry.example.com/execd:1"}],
        "containers": containers,
        "volumes": [{"name": "opensandbox-bin", "emptyDir": {}}],
    }


def test_services_are_placed_by_role_after_egress() -> None:
    pod_spec = stock_pod_spec()

    add_services(pod_spec, SERVICES, "Always")

    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "egress", "db", "migrate", "cache"]
    assert [c["name"] for c in pod_spec["containers"]] == ["sandbox", "worker"]
    init = {c["name"]: c for c in pod_spec["initContainers"]}
    assert init["egress"]["restartPolicy"] == "Always"
    assert init["egress"]["startupProbe"]["httpGet"]["port"] == 18080

    db = init["db"]
    assert db["restartPolicy"] == "Always"
    assert db["imagePullPolicy"] == "Always"
    assert db["startupProbe"]["exec"]["command"] == ["pg_isready"]
    assert db["readinessProbe"]["failureThreshold"] == 3
    assert db["env"] == [{"name": "POSTGRES_PASSWORD", "value": "pw"}]
    assert db["volumeMounts"] == [
        {"name": "nemo-vol-shared", "mountPath": "/data", "readOnly": False},
        {"name": "nemo-shm-db", "mountPath": "/dev/shm"},
    ]
    assert db["securityContext"]["capabilities"]["drop"] == ["NET_ADMIN", "NET_RAW"]

    migrate = init["migrate"]
    assert migrate["command"] == ["true"]
    assert not {"restartPolicy", "startupProbe", "readinessProbe"} & migrate.keys()

    # A sidecar without readiness only holds the next container until it starts.
    assert init["cache"]["restartPolicy"] == "Always"
    assert "startupProbe" not in init["cache"]

    worker = pod_spec["containers"][1]
    assert "restartPolicy" not in worker and "startupProbe" not in worker
    assert worker["readinessProbe"]["exec"]["command"] == ["redis-cli", "ping"]
    assert worker["resources"] == {"limits": {"cpu": "1"}}


def test_pod_gets_volumes_host_aliases_and_main_options() -> None:
    pod_spec = stock_pod_spec()

    add_services(pod_spec, SERVICES, "IfNotPresent")

    assert pod_spec["volumes"][1:] == [
        {"name": "nemo-vol-shared", "emptyDir": {}},
        {"name": "nemo-shm-db", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}},
    ]
    assert pod_spec["hostAliases"] == [{"ip": "127.0.0.1", "hostnames": ["db", "migrate", "cache", "worker", "main"]}]
    sandbox = pod_spec["containers"][0]
    assert sandbox["volumeMounts"] == [{"name": "nemo-vol-shared", "mountPath": "/shared", "readOnly": False}]
    assert sandbox["securityContext"]["capabilities"] == {"drop": ["NET_ADMIN"], "add": ["SYS_PTRACE"]}


def test_sandbox_without_network_policy_has_no_egress_to_move() -> None:
    pod_spec = stock_pod_spec(egress=False)

    add_services(pod_spec, SERVICES, "IfNotPresent")

    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "db", "migrate", "cache"]


@pytest.mark.parametrize(
    ("pod_spec", "message"),
    [
        ({"containers": [{"name": "other"}]}, "first container is not 'sandbox'"),
        ({"containers": [{"name": "sandbox"}, {"name": "db"}]}, r"clash .* \['db'\]"),
    ],
)
def test_unexpected_pod_shape_is_rejected(pod_spec: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        add_services(pod_spec, SERVICES, "IfNotPresent")


def _status(name: str, *, ready: bool = False, restarts: int = 0, **state: Any) -> V1ContainerStatus:
    """A container status for ``name``; ``state`` is one ``V1ContainerState`` field (running, waiting, terminated)."""
    return V1ContainerStatus(
        name=name, image="i", image_id="", ready=ready, restart_count=restarts, state=V1ContainerState(**state)
    )


def _running(name: str, *, ready: bool = True, restarts: int = 0) -> V1ContainerStatus:
    """A running container, ready unless told otherwise."""
    return _status(name, ready=ready, restarts=restarts, running=V1ContainerStateRunning())


def _exited(name: str, code: int, *, restarts: int = 0) -> V1ContainerStatus:
    """A container that terminated with exit ``code``."""
    return _status(name, restarts=restarts, terminated=V1ContainerStateTerminated(exit_code=code, reason="Error"))


def _pod(*statuses: V1ContainerStatus) -> V1Pod:
    """A pod with these statuses, split the way Kubernetes reports them: regular containers vs init containers."""
    # sandbox and the after_main service are regular containers; every other service is an init container.
    main = {"sandbox", "worker"}
    return V1Pod(
        status=V1PodStatus(
            init_container_statuses=[s for s in statuses if s.name not in main],
            container_statuses=[s for s in statuses if s.name in main],
        )
    )


def _states(*statuses: V1ContainerStatus, services: ComposeServices = SERVICES) -> tuple[Any, list[str]]:
    """``service_states`` for a pod with these container statuses."""
    return service_states(_pod(*statuses), services)


def test_all_services_up() -> None:
    assert _states(_running("db"), _exited("migrate", 0), _running("cache"), _running("worker")) == (None, [])


def test_services_still_starting_are_not_ready() -> None:
    assert _states(_running("db", ready=False)) == (None, ["db", "migrate", "cache", "worker"])
    assert _states(_running("db"), _status("migrate", running=V1ContainerStateRunning())) == (
        None,
        ["migrate", "cache", "worker"],
    )


@pytest.mark.parametrize(
    ("statuses", "failure"),
    [
        ((_running("db"), _exited("migrate", 2)), ("migrate", "exited with code 2 (Error)")),
        (
            (_status("db", waiting=V1ContainerStateWaiting(reason="ImagePullBackOff", message="not found")),),
            ("db", "ImagePullBackOff: not found"),
        ),
        (
            (_running("db"), _exited("migrate", 0), _running("cache"), _exited("worker", 0)),
            ("worker", "exited with code 0 (Error)"),
        ),
        ((_running("db", restarts=2),), None),
    ],
)
def test_failures(statuses: tuple[V1ContainerStatus, ...], failure: tuple[str, str] | None) -> None:
    assert _states(*statuses)[0] == failure


@pytest.mark.parametrize(("restarts", "failed"), [(2, False), (3, True)])
def test_crash_looping_sidecar_fails_after_three_restarts(restarts: int, failed: bool) -> None:
    crashing = V1ContainerStatus(
        name="db",
        image="i",
        image_id="",
        ready=False,
        restart_count=restarts,
        state=V1ContainerState(waiting=V1ContainerStateWaiting(reason="CrashLoopBackOff")),
        last_state=V1ContainerState(terminated=V1ContainerStateTerminated(exit_code=1, reason="Error")),
    )

    assert _states(crashing)[0] == (("db", "exited with code 1 (Error)") if failed else None)
