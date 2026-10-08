# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pod builder: how services are added to the stock BatchSandbox pod, and how their state is read back."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import nemo_ext_pod as pod_builder
import pytest
from _ext_doubles import (
    SPEC_ROLES,
    all_ready,
    by_name,
    pod,
    running,
    spec_dict,
    stock_batchsandbox,
    terminated,
    waiting,
)
from kubernetes.client import V1ContainerStatus, V1ObjectMeta, V1Pod
from nemo_ext_spec import ServicesSpec, ServicesSpecError


def inject(spec: dict[str, Any], *, egress: bool = True) -> dict[str, Any]:
    """The stock manifest after ``inject_services`` added ``spec``'s services."""
    body = stock_batchsandbox(egress=egress)
    pod_builder.inject_services(body, ServicesSpec.model_validate(spec), "IfNotPresent")
    return body


# ---- pod spec ----------------------------------------------------------------------------


def test_services_are_added_to_the_stock_batchsandbox() -> None:
    template = inject(spec_dict())["spec"]["template"]
    pod_spec = template["spec"]
    init = by_name(pod_spec["initContainers"])
    containers = by_name(pod_spec["containers"])

    # Egress starts first so its rules cover every service; the run-once job runs in order.
    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "egress", "db", "migrate", "proxy"]
    assert [c["name"] for c in pod_spec["containers"]] == ["sandbox", "worker"]

    egress = init["egress"]
    assert egress["restartPolicy"] == "Always"
    assert egress["startupProbe"]["failureThreshold"] == pod_builder.EGRESS_STARTUP_PROBE_FAILURES
    # proxy claims 18080, so the egress policy server moves.
    assert {"name": pod_builder.EGRESS_ADDR_ENV, "value": ":18090"} in egress["env"]
    assert egress["ports"] == [{"name": "egress-api", "containerPort": 18090}]
    assert egress["readinessProbe"]["httpGet"]["port"] == 18090

    db = init["db"]
    assert db["restartPolicy"] == "Always"
    assert db["startupProbe"]["exec"]["command"] == ["pg_isready", "-U", "postgres"]
    assert db["startupProbe"]["failureThreshold"] == pod_builder.STARTUP_PROBE_FAILURES
    assert db["readinessProbe"]["failureThreshold"] == pod_builder.READINESS_PROBE_FAILURES
    assert db["env"] == [{"name": "POSTGRES_PASSWORD", "value": "pw"}]
    assert db["ports"] == [{"containerPort": 5432}]
    assert db["volumeMounts"] == [{"name": "nemo-vol-shared", "mountPath": "/data", "readOnly": False}]
    assert db["securityContext"] == {"privileged": False, "capabilities": {"drop": ["NET_ADMIN", "NET_RAW"]}}
    assert db["imagePullPolicy"] == "IfNotPresent"

    migrate = init["migrate"]
    assert "restartPolicy" not in migrate and "readinessProbe" not in migrate
    assert migrate["command"] == ["true"]

    assert {"name": "nemo-shm-proxy", "mountPath": "/dev/shm"} in init["proxy"]["volumeMounts"]

    worker = containers["worker"]
    assert "startupProbe" not in worker and "restartPolicy" not in worker
    assert worker["readinessProbe"]["exec"]["command"] == ["redis-cli", "ping"]

    main = containers["sandbox"]
    assert {"name": "nemo-vol-shared", "mountPath": "/shared", "readOnly": False} in main["volumeMounts"]
    assert main["securityContext"]["capabilities"] == {"drop": ["NET_ADMIN"], "add": ["SYS_PTRACE"]}

    volumes = by_name(pod_spec["volumes"])
    assert volumes["nemo-vol-shared"] == {"name": "nemo-vol-shared", "emptyDir": {}}
    assert volumes["nemo-shm-proxy"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "64Mi"}
    assert "opensandbox-bin" in volumes

    assert pod_spec["hostAliases"] == [{"ip": "127.0.0.1", "hostnames": ["db", "migrate", "proxy", "worker", "main"]}]
    assert json.loads(template["metadata"]["annotations"][pod_builder.ROLES_ANNOTATION]) == SPEC_ROLES


def test_egress_keeps_its_port_when_no_service_claims_it() -> None:
    spec = spec_dict()
    spec["services"][2]["ports"] = [8080]

    egress = by_name(inject(spec)["spec"]["template"]["spec"]["initContainers"])["egress"]

    assert not any(e["name"] == pod_builder.EGRESS_ADDR_ENV for e in egress["env"])
    assert egress["readinessProbe"]["httpGet"]["port"] == pod_builder.EGRESS_DEFAULT_PORT


def test_services_without_network_policy() -> None:
    names = [c["name"] for c in inject(spec_dict(), egress=False)["spec"]["template"]["spec"]["initContainers"]]
    assert names == ["execd-installer", "db", "migrate", "proxy"]


def test_a_sidecar_without_readiness_has_no_probes() -> None:
    spec = spec_dict()
    del spec["services"][0]["readiness"]

    db = by_name(inject(spec)["spec"]["template"]["spec"]["initContainers"])["db"]

    assert db["restartPolicy"] == "Always"
    assert "startupProbe" not in db and "readinessProbe" not in db


def test_an_unexpected_stock_pod_is_rejected() -> None:
    body = stock_batchsandbox()
    containers = body["spec"]["template"]["spec"]["containers"]
    containers.insert(0, containers.pop())

    with pytest.raises(ServicesSpecError, match="first container is not 'sandbox'"):
        pod_builder.inject_services(body, ServicesSpec.model_validate(spec_dict()), "IfNotPresent")


def test_a_service_cannot_shadow_a_stock_container() -> None:
    body = stock_batchsandbox()
    body["spec"]["template"]["spec"]["initContainers"].append({"name": "db", "image": "stock"})

    with pytest.raises(ServicesSpecError, match=r"clash with sandbox containers: \['db'\]"):
        pod_builder.inject_services(body, ServicesSpec.model_validate(spec_dict()), "IfNotPresent")


# ---- pod status --------------------------------------------------------------------------


def test_services_waiting_lists_what_is_not_up_in_start_order() -> None:
    starting = pod([running("egress"), running("db", ready=False)], [])
    assert pod_builder.services_waiting(starting, SPEC_ROLES) == ["db", "migrate", "proxy", "worker", "main"]

    almost = pod(
        [running("egress"), running("db"), terminated("migrate", 0), running("proxy")],
        [running("sandbox"), running("worker", ready=False)],
    )
    assert pod_builder.services_waiting(almost, SPEC_ROLES) == ["worker"]

    assert pod_builder.services_waiting(all_ready(), SPEC_ROLES) == []


def test_a_run_once_service_that_failed_is_still_waited_for() -> None:
    failed = pod([running("egress"), running("db"), terminated("migrate", 1)], [])
    assert "migrate" in pod_builder.services_waiting(failed, SPEC_ROLES)


@pytest.mark.parametrize(
    ("init", "containers", "phase", "failed", "detail"),
    [
        # A job that exits non-zero fails the pod.
        ([running("egress"), running("db"), terminated("migrate", 3)], [], "Failed", "migrate", "exit code 3, Error"),
        # A sidecar that keeps crashing.
        ([running("egress"), terminated("db", 1, restarts=3)], [], "Pending", "db", "exit code 1, Error"),
        # An image that can't be pulled never clears on its own.
        (
            [running("egress"), waiting("db", "ImagePullBackOff")],
            [],
            "Pending",
            "db",
            "ImagePullBackOff: no such image",
        ),
        # A service next to main is never restarted, so any exit is final.
        (
            [running("egress"), running("db"), terminated("migrate", 0), running("proxy")],
            [running("sandbox"), terminated("worker", 0)],
            "Running",
            "worker",
            "exit code 0, Error",
        ),
    ],
)
def test_service_failure_names_the_failed_service(
    init: list[V1ContainerStatus], containers: list[V1ContainerStatus], phase: str, failed: str, detail: str
) -> None:
    assert pod_builder.service_failure(pod(init, containers, phase), SPEC_ROLES) == pod_builder.ServiceFailure(
        failed, detail
    )


def test_a_sidecar_restart_below_the_threshold_is_not_a_failure() -> None:
    restarting = pod([running("egress"), terminated("db", 1, restarts=1)], [])
    assert pod_builder.service_failure(restarting, SPEC_ROLES) is None


def test_the_first_exit_is_reported_when_several_services_failed() -> None:
    # Once migrate fails the pod, Kubernetes stops db too; migrate is the cause.
    early, late = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC), datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    failed = pod(
        [
            terminated("db", 137, finished_at=late),
            terminated("migrate", 2, finished_at=early),
            waiting("proxy", "ErrImageNeverPull"),
        ],
        [],
        "Failed",
    )
    assert pod_builder.service_failure(failed, SPEC_ROLES) == pod_builder.ServiceFailure(
        "migrate", "exit code 2, Error"
    )


def test_a_failed_pod_without_a_failed_service_is_blamed_on_main() -> None:
    evicted = pod([running("egress")], [], "Failed")
    evicted.status.reason = "Evicted"
    assert pod_builder.service_failure(evicted, SPEC_ROLES) == pod_builder.ServiceFailure(
        "sandbox", "pod failed: Evicted"
    )


@pytest.mark.parametrize("annotation", [None, "not json", '["db"]'])
def test_pod_roles_of_a_pod_without_services(annotation: str | None) -> None:
    annotations = {pod_builder.ROLES_ANNOTATION: annotation} if annotation is not None else None
    assert pod_builder.pod_roles(V1Pod(metadata=V1ObjectMeta(annotations=annotations))) == {}


def test_pod_roles_reads_back_what_inject_recorded() -> None:
    assert pod_builder.pod_roles(all_ready()) == SPEC_ROLES
