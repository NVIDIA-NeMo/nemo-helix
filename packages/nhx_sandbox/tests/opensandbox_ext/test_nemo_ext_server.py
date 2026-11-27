# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox server extension against the real opensandbox-server provider, with the Kubernetes API faked."""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opensandbox_server")

import nemo_ext_server as ext
import opensandbox_server
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kubernetes.client import (
    V1ContainerState,
    V1ContainerStateTerminated,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodStatus,
)
from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_EXTENSION_KEY,
    AGENT_BUNDLE_POD_ANNOTATION,
    EGRESS_API_PORT_ANNOTATION,
    EXTENSION_KEY,
    OOM_KILL_PROCESS_POD_ANNOTATION,
    POD_ANNOTATION,
)
from opensandbox_server.api.schema import ImageSpec, NetworkPolicy, NetworkRule
from opensandbox_server.extensions import apply_extensions_to_annotations
from opensandbox_server.services.constants import SANDBOX_EGRESS_AUTH_TOKEN_METADATA_KEY, SandboxErrorCodes
from opensandbox_server.services.endpoint_auth import build_egress_auth_headers
from opensandbox_server.services.k8s import kubernetes_service, provider_factory
from opensandbox_server.services.k8s.batchsandbox_provider import BatchSandboxProvider
from opensandbox_server.services.k8s.kubernetes_service import KubernetesSandboxService

SANDBOX_ID = "sbx-1"
SPEC = (
    '{"services":[{"name":"db","image":"docker.io/library/postgres:16","ports":[5432],'
    '"readiness":{"exec":["pg_isready"]}}],"main":{"add_capabilities":["SYS_PTRACE"]}}'
)
BUNDLE_SPEC = (
    '{"image":"registry.example.com/bundles/claude@sha256:' + "a" * 64 + '",'
    '"installer":"/opt/agent-bundle/copy-agent","entrypoint":"bin/claude"}'
)
EXT_DIR = Path(__file__).resolve().parents[2] / "opensandbox_ext"
SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class FakeK8sClient:
    """The server's ``K8sClient``: records created manifests and returns the pods a test sets."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.pods: list[V1Pod] = []
        # The namespace of each pod lookup, to see which one a route searched.
        self.listed_namespaces: list[str] = []
        self.workloads: list[dict[str, Any]] = []

    def get_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> Any:
        return next((w for w in self.workloads if w["metadata"]["labels"]["opensandbox.io/id"] == name), None)

    def list_custom_objects(
        self, group: str, version: str, namespace: str, plural: str, label_selector: str
    ) -> list[dict[str, Any]]:
        return self.workloads

    def create_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict[str, Any]):
        self.created.append(body)
        return {"metadata": {"name": body["metadata"]["name"], "uid": "uid-1"}}

    def list_pods(self, namespace: str, label_selector: str = "") -> list[V1Pod]:
        assert label_selector == f"opensandbox.io/id={SANDBOX_ID}"
        self.listed_namespaces.append(namespace)
        return self.pods

    def get_core_v1_api(self) -> Any:
        return types.SimpleNamespace(read_namespaced_pod_log=lambda *a, **k: "FATAL: bad config")


@pytest.fixture
def k8s() -> FakeK8sClient:
    """A fresh fake Kubernetes client per test."""
    return FakeK8sClient()


@pytest.fixture
def provider(k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch) -> ext.NemoComposeProvider:
    """The extension's provider on the fake client; it registers itself as the routes' provider."""
    # Reset the module-level provider so one test's provider can't leak into the next.
    monkeypatch.setattr(ext, "_provider", None)
    return ext.NemoComposeProvider(k8s)


# The egress policy every test sandbox gets unless a test passes its own.
EGRESS_POLICY = NetworkPolicy(defaultAction="deny", egress=[NetworkRule(action="allow", target="pypi.org")])


def _create(
    provider: BatchSandboxProvider,
    annotations: dict[str, str] | None,
    network_policy: NetworkPolicy | None = EGRESS_POLICY,
    **kwargs: Any,
) -> None:
    """The stock create path for the test sandbox, with an egress policy unless ``network_policy`` is None."""
    provider.create_workload(
        sandbox_id=SANDBOX_ID,
        namespace="nemo",
        image_spec=ImageSpec(uri="registry.example.com/task:1"),
        entrypoint=["/app/entrypoint.sh"],
        env={},
        resource_limits={"cpu": "1", "memory": "1Gi"},
        labels={"opensandbox.io/id": SANDBOX_ID},
        expires_at=None,
        execd_image="registry.example.com/execd:1",
        network_policy=network_policy,
        egress_image="registry.example.com/egress:1",
        annotations=annotations,
        **kwargs,
    )


def test_stock_server_turns_the_extensions_into_pod_annotations() -> None:
    annotations: dict[str, str] = {}
    apply_extensions_to_annotations(annotations, {EXTENSION_KEY: SPEC, AGENT_BUNDLE_EXTENSION_KEY: BUNDLE_SPEC})
    assert annotations == {POD_ANNOTATION: SPEC, AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}


def test_services_are_added_to_the_stock_manifest(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    _create(provider, {POD_ANNOTATION: SPEC})

    pod_spec = k8s.created[0]["spec"]["template"]["spec"]
    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "egress", "db"]
    assert [c["name"] for c in pod_spec["containers"]] == ["sandbox"]
    assert pod_spec["containers"][0]["securityContext"]["capabilities"]["add"] == ["SYS_PTRACE"]
    assert k8s.created[0]["spec"]["template"]["metadata"]["annotations"][POD_ANNOTATION] == SPEC


def test_requests_without_services_are_untouched(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    stock = FakeK8sClient()
    _create(BatchSandboxProvider(stock), None)  # ty: ignore[invalid-argument-type]
    _create(provider, None)
    assert k8s.created == stock.created


def test_invalid_spec_is_a_value_error_and_creates_nothing(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    with pytest.raises(ValueError, match=r"ports \[44772\] are used"):
        _create(provider, {POD_ANNOTATION: SPEC.replace("5432", "44772")})
    assert k8s.created == []


@pytest.mark.parametrize("annotation", [{POD_ANNOTATION: SPEC}, {AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}])
def test_pool_mode_is_rejected(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, annotation: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="can't be combined with poolRef"):
        _create(provider, annotation, extensions={"poolRef": "pool-1"})
    assert k8s.created == []


def test_bundle_is_added_to_the_stock_manifest_before_the_services(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    _create(provider, {AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC, POD_ANNOTATION: SPEC})

    pod_spec = k8s.created[0]["spec"]["template"]["spec"]
    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "egress", "agent-bundle", "db"]
    bundle = pod_spec["initContainers"][2]
    assert bundle["image"] == "registry.example.com/bundles/claude@sha256:" + "a" * 64
    assert bundle["command"] == ["/opt/agent-bundle/copy-agent", "/installed-agent"]
    sandbox_mounts = pod_spec["containers"][0]["volumeMounts"]
    assert {"name": "nemo-installed-agent", "mountPath": "/installed-agent", "readOnly": True} in sandbox_mounts


@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.parametrize("server_setting", ["true", None])
def test_an_oom_kill_request_annotates_the_pod_only_on_a_server_that_enables_it(
    provider: ext.NemoComposeProvider,
    k8s: FakeK8sClient,
    monkeypatch: pytest.MonkeyPatch,
    requested: bool,
    server_setting: str | None,
) -> None:
    """A node plugin that doesn't allow memory.oom.group rejects the container, so only the operator can turn it on."""
    if server_setting is None:
        monkeypatch.delenv(ext.OOM_KILL_PROCESS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ext.OOM_KILL_PROCESS_ENV_VAR, server_setting)

    _create(provider, {OOM_KILL_PROCESS_POD_ANNOTATION: "true"} if requested else None)

    annotations = k8s.created[0]["spec"]["template"]["metadata"].get("annotations") or {}
    assert ("memory.oom.group.memory-qos.nri.io/sandbox" in annotations) is (requested and server_setting == "true")


def test_invalid_bundle_is_a_value_error_and_creates_nothing(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    with pytest.raises(ValueError, match="image"):
        # No registry, which CRI-O won't resolve.
        _create(provider, {AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC.replace("registry.example.com/", "")})
    assert k8s.created == []


def _workload(annotations: dict[str, str], phase: str, pod_ip: str | None) -> dict[str, Any]:
    """A BatchSandbox object as the server reads it back: pod template annotations, phase, and pod IP if known."""
    metadata: dict[str, Any] = {"name": "sbx-1", "annotations": {}}
    # The controller publishes the pod IP as this annotation, which is what _parse_pod_ip reads.
    if pod_ip is not None:
        metadata["annotations"]["sandbox.opensandbox.io/endpoints"] = f'["{pod_ip}"]'
    return {
        "metadata": metadata,
        "spec": {"template": {"metadata": {"annotations": annotations}}},
        "status": {"phase": phase, "allocated": 1, "replicas": 1, "ready": 0},
    }


@pytest.mark.parametrize(
    ("annotations", "phase", "pod_ip", "state"),
    [
        ({POD_ANNOTATION: SPEC}, "Pending", "10.0.0.5", "Allocated"),
        # The client looks up the sandbox's endpoints right after create, which needs the IP.
        ({POD_ANNOTATION: SPEC}, "Pending", None, "Pending"),
        ({POD_ANNOTATION: SPEC}, "Running", "10.0.0.5", "Running"),
        ({POD_ANNOTATION: SPEC}, "Failed", "10.0.0.5", "Failed"),
        ({AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}, "Pending", "10.0.0.5", "Allocated"),
        ({AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}, "Pending", None, "Pending"),
        ({AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}, "Running", "10.0.0.5", "Running"),
        ({}, "Pending", "10.0.0.5", "Pending"),
    ],
)
def test_pod_with_services_or_a_bundle_and_an_ip_counts_as_allocated(
    provider: ext.NemoComposeProvider, annotations: dict[str, str], phase: str, pod_ip: str | None, state: str
) -> None:
    # The stock create returns on Allocated, so it doesn't block until every service is up.
    assert provider.get_status(_workload(annotations, phase, pod_ip))["state"] == state


def _labelled_workload(annotations: dict[str, str], phase: str, pod_ip: str | None) -> dict[str, Any]:
    """``_workload`` with the sandbox ID label the server puts on every BatchSandbox it creates."""
    workload = _workload(annotations, phase, pod_ip)
    workload["metadata"]["labels"] = {"opensandbox.io/id": SANDBOX_ID}
    workload["metadata"]["namespace"] = "nemo"
    return workload


def _status(provider: ext.NemoComposeProvider, k8s: FakeK8sClient, workload: dict[str, Any]) -> dict[str, Any]:
    """The sandbox's status as the server reads it: the BatchSandbox fetched, then its status derived."""
    k8s.workloads = [workload]
    fetched = provider.get_workload(SANDBOX_ID, "nemo")
    assert fetched is not None
    return provider.get_status(fetched)


def _pod(*containers: V1ContainerStatus) -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(name="sbx-1-pod", namespace="nemo"),
        status=V1PodStatus(container_statuses=list(containers)),
    )


def _container(
    name: str, terminated: V1ContainerStateTerminated | None, waiting: V1ContainerStateWaiting | None = None
) -> V1ContainerStatus:
    state = V1ContainerState(terminated=terminated, waiting=waiting)
    return V1ContainerStatus(name=name, state=state, image="i", image_id="i", ready=False, restart_count=0)


@pytest.mark.parametrize(
    ("annotations", "pod_ip"),
    [
        ({}, None),
        ({}, "10.0.0.5"),
        ({POD_ANNOTATION: SPEC}, "10.0.0.5"),
        ({AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}, "10.0.0.5"),
    ],
)
def test_pending_sandbox_whose_container_stopped_is_failed_with_its_reason(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, annotations: dict[str, str], pod_ip: str | None
) -> None:
    """The controller leaves a dead sandbox Pending, which the client would report as still being created.

    With services or a bundle the pod keeps its IP, so without this a dead sandbox reads as Allocated forever.
    """
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]

    status = _status(provider, k8s, _labelled_workload(annotations, "Pending", pod_ip))

    assert status["state"] == "Failed"
    assert status["reason"] == "CONTAINER_TERMINATED"
    assert status["message"] == "container sandbox: OOMKilled (exit 137)"
    assert k8s.listed_namespaces == ["nemo"]


def test_allocated_sandbox_whose_container_stopped_is_failed_with_its_reason(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    """A BatchSandbox with no phase and a pod IP reads as Allocated, which would end the create wait as a success."""
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]

    status = _status(provider, k8s, _labelled_workload({}, "", "10.0.0.5"))

    assert (status["state"], status["message"]) == ("Failed", "container sandbox: OOMKilled (exit 137)")


def test_a_service_containers_oom_kill_is_reported_under_its_own_name(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    """The client tells the agent's sandbox running out of memory from a compose service doing so by this name."""
    k8s.pods = [
        _pod(
            _container("sandbox", None),
            _container("db", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")),
        )
    ]

    status = _status(provider, k8s, _labelled_workload({POD_ANNOTATION: SPEC}, "Pending", "10.0.0.5"))

    assert (status["state"], status["message"]) == ("Failed", "container db: OOMKilled (exit 137)")


def test_the_sandbox_containers_stop_is_reported_before_a_service_that_stopped_with_it(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    """An ``after_main`` service that talks to the agent stops when the agent's container does.

    The kubelet lists containers by name, so ``api`` comes before ``sandbox``; reporting it first would hide the
    sandbox's OOM kill behind the service's follow-on failure.
    """
    k8s.pods = [
        _pod(
            _container("api", V1ContainerStateTerminated(exit_code=1, reason="Error")),
            _container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")),
        )
    ]

    status = _status(provider, k8s, _labelled_workload({POD_ANNOTATION: SPEC}, "Pending", "10.0.0.5"))

    assert (status["state"], status["message"]) == ("Failed", "container sandbox: OOMKilled (exit 137)")


@pytest.mark.parametrize(("name", "state"), [("db-init", "Allocated"), ("sandbox", "Failed")])
def test_only_the_sandbox_container_has_failed_when_it_exits_0(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, name: str, state: str
) -> None:
    """A service container that finished its work exits 0; the sandbox container is never meant to stop."""
    k8s.pods = [_pod(_container(name, V1ContainerStateTerminated(exit_code=0, reason="Completed")))]

    assert _status(provider, k8s, _labelled_workload({POD_ANNOTATION: SPEC}, "Pending", "10.0.0.5"))["state"] == state


def test_pending_sandbox_whose_containers_run_keeps_the_stock_status(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    k8s.pods = [_pod(_container("sandbox", None))]

    assert _status(provider, k8s, _labelled_workload({}, "Pending", None))["state"] == "Pending"
    assert _status(provider, k8s, _labelled_workload({POD_ANNOTATION: SPEC}, "Pending", "10.0.0.5"))["state"] == (
        "Allocated"
    )


REFUSED_MESSAGE = 'CreateContainer nemo/sbx-1-pod:sandbox: invalid annotation: "memory.oom.group"'


@pytest.mark.parametrize(("annotations", "pod_ip"), [({}, None), ({POD_ANNOTATION: SPEC}, "10.0.0.5")])
def test_pending_sandbox_whose_container_the_runtime_refused_is_failed_with_why(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, annotations: dict[str, str], pod_ip: str | None
) -> None:
    """An NRI plugin that doesn't allow an annotation refuses the container for good, so the sandbox never starts."""
    waiting = V1ContainerStateWaiting(reason="CreateContainerError", message=REFUSED_MESSAGE)
    k8s.pods = [_pod(_container("sandbox", None, waiting))]

    status = _status(provider, k8s, _labelled_workload(annotations, "Pending", pod_ip))

    assert status["state"] == "Failed"
    assert status["reason"] == "CONTAINER_REFUSED"
    assert status["message"] == f"container sandbox: CreateContainerError: {REFUSED_MESSAGE}"


@pytest.mark.parametrize(
    "waiting",
    [
        V1ContainerStateWaiting(reason="ContainerCreating"),
        # Other create errors, such as a name still held by a removed container, clear up on the kubelet's retry.
        V1ContainerStateWaiting(reason="CreateContainerError", message='the container name "sandbox" is reserved'),
    ],
)
def test_pending_sandbox_whose_container_waits_otherwise_keeps_the_stock_status(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, waiting: V1ContainerStateWaiting
) -> None:
    k8s.pods = [_pod(_container("sandbox", None, waiting))]

    assert _status(provider, k8s, _labelled_workload({}, "Pending", None))["state"] == "Pending"


async def test_stock_create_wait_ends_with_the_message_of_a_sandbox_that_cannot_start(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create would otherwise wait out the server's create timeout (900 s in our examples) and say only that."""
    waiting = V1ContainerStateWaiting(reason="CreateContainerError", message=REFUSED_MESSAGE)
    k8s.pods = [_pod(_container("sandbox", None, waiting))]
    k8s.workloads = [_labelled_workload({}, "Pending", None)]
    monkeypatch.setattr(kubernetes_service, "_is_unschedulable_status", ext.end_create_wait)
    service = types.SimpleNamespace(workload_provider=provider, namespace="nemo")

    with pytest.raises(HTTPException) as raised:
        await KubernetesSandboxService._wait_for_sandbox_ready(
            service,  # ty: ignore[invalid-argument-type]
            SANDBOX_ID,
            timeout_seconds=5,
            poll_interval_seconds=0.01,
        )

    assert raised.value.detail == {
        "code": SandboxErrorCodes.K8S_POD_FAILED,
        "message": f"Sandbox failed to start: container sandbox: CreateContainerError: {REFUSED_MESSAGE}",
    }


@pytest.mark.parametrize(
    ("state", "reason", "unschedulable"),
    [
        # The stock provider reports an unschedulable sandbox as Failed; the stock wait answers it with a 400.
        ("Failed", "POD_PLATFORM_UNSCHEDULABLE", True),
        ("Pending", "POD_PENDING", False),
    ],
)
def test_create_wait_keeps_the_stock_check_for_a_sandbox_that_has_not_failed(
    state: str, reason: str, unschedulable: bool
) -> None:
    assert ext.end_create_wait({"state": state, "reason": reason, "message": ""}) is unschedulable


def test_status_reads_no_pod_on_the_server_event_loop(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    """The stock create wait calls get_status on the event loop, where a pod read would stall every other request."""
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]

    assert provider.get_status(_labelled_workload({}, "Pending", None))["state"] == "Pending"
    assert k8s.listed_namespaces == []


def test_listed_sandbox_whose_container_stopped_is_failed_with_its_reason(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]
    k8s.workloads = [_labelled_workload({}, "Pending", None)]

    (listed,) = provider.list_workloads("nemo", "opensandbox.io/id")

    assert provider.get_status(listed)["message"] == "container sandbox: OOMKilled (exit 137)"


def test_a_failed_sandbox_leaves_the_fetched_workload_as_it_was(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    """With the server's informer on, a fetched workload is its cached object, which every later read shares.

    The fake client hands back the workloads it stores, as the informer does.
    """
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]
    workload = _labelled_workload({}, "Pending", None)
    k8s.workloads = [workload]

    fetched = provider.get_workload(SANDBOX_ID, "nemo")
    (listed,) = provider.list_workloads("nemo", "opensandbox.io/id")

    assert fetched is not None
    assert provider.get_status(fetched)["state"] == provider.get_status(listed)["state"] == "Failed"
    assert workload == _labelled_workload({}, "Pending", None)
    # A later read reports the pod as it is then, not as an earlier read found it.
    k8s.pods = [_pod(_container("sandbox", None))]
    assert _status(provider, k8s, workload)["state"] == "Pending"


def test_running_sandbox_is_not_looked_up(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    """Every status read fetches the workload, so a pod lookup is spent only on a sandbox that isn't running yet."""
    k8s.pods = [_pod(_container("sandbox", V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")))]

    assert _status(provider, k8s, _labelled_workload({}, "Running", "10.0.0.5"))["state"] == "Running"
    assert k8s.listed_namespaces == []


def test_unscheduled_sandbox_is_not_looked_up(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    """The create wait reads the status every second, and a sandbox waiting for a node has no containers to report."""
    workload = _labelled_workload({}, "Pending", None)
    workload["status"]["allocated"] = 0

    assert _status(provider, k8s, workload)["state"] == "Pending"
    assert k8s.listed_namespaces == []


def test_failed_pod_lookup_keeps_the_stock_status(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_: Any) -> list[V1Pod]:
        raise RuntimeError("API server unavailable")

    monkeypatch.setattr(k8s, "list_pods", fail)

    assert _status(provider, k8s, _labelled_workload({}, "Pending", None))["state"] == "Pending"


async def test_stock_create_wait_returns_for_a_pending_bundle_only_sandbox(
    provider: ext.NemoComposeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bundle image that can't be pulled leaves the pod Pending with an IP. The stock create must
    # return anyway, so the client's wait reports the bundle failure instead of a create timeout.
    workload = _workload({AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC}, "Pending", "10.0.0.5")
    monkeypatch.setattr(provider, "get_workload", lambda **_: workload)
    service = types.SimpleNamespace(workload_provider=provider, namespace="nemo")

    returned = await KubernetesSandboxService._wait_for_sandbox_ready(
        service,  # ty: ignore[invalid-argument-type]
        SANDBOX_ID,
        timeout_seconds=2,
        poll_interval_seconds=0.01,
    )

    assert returned is workload


@pytest.fixture
def api(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> TestClient:
    """A test client for the extension's routes, with one sandbox pod that carries the test spec."""
    k8s.pods = [
        V1Pod(
            metadata=V1ObjectMeta(name="sbx-1-pod", namespace="nemo", annotations={POD_ANNOTATION: SPEC}),
            status=V1PodStatus(init_container_statuses=[], container_statuses=[]),
        )
    ]
    app = FastAPI()
    app.include_router(ext.router)
    return TestClient(app)


def test_health(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert api.get("/v1/nemo-ext/health").json() == {
        "status": "ok",
        "extension": "nemo-compose-services",
        "features": [
            "compose-services",
            "agent-bundle",
            "compose-main",
            "compose-security",
            "egress-port",
        ],
    }
    monkeypatch.setattr(ext, "_provider", None)
    assert api.get("/v1/nemo-ext/health").status_code == 503


def test_services_status(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").json() == {"failure": None, "not_ready": ["db"]}

    monkeypatch.setattr(ext, "service_states", lambda pod, services: (("db", "exited with code 1 (Error)"), ["db"]))
    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").json() == {
        "failure": "service 'db' exited with code 1 (Error). Last log lines:\nFATAL: bad config",
        "not_ready": ["db"],
    }


def _bundle_pod(k8s: FakeK8sClient, *init_statuses: V1ContainerStatus, services: bool = False) -> None:
    """Make the test pod carry the agent bundle, and the services if asked, with these init container statuses."""
    annotations = {AGENT_BUNDLE_POD_ANNOTATION: BUNDLE_SPEC, **({POD_ANNOTATION: SPEC} if services else {})}
    k8s.pods[0].metadata.annotations = annotations
    k8s.pods[0].status = V1PodStatus(init_container_statuses=list(init_statuses), container_statuses=[])


def _bundle_status(name: str, **state: Any) -> V1ContainerStatus:
    return V1ContainerStatus(
        name=name, image="i", image_id="", ready=False, restart_count=0, state=V1ContainerState(**state)
    )


def test_bundle_status(api: TestClient, k8s: FakeK8sClient) -> None:
    url = f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services"
    _bundle_pod(k8s)
    assert api.get(url).json() == {"failure": None, "not_ready": ["agent-bundle"]}

    _bundle_pod(k8s, _bundle_status("agent-bundle", terminated=V1ContainerStateTerminated(exit_code=0)))
    assert api.get(url).json() == {"failure": None, "not_ready": []}

    _bundle_pod(k8s, _bundle_status("agent-bundle", terminated=V1ContainerStateTerminated(exit_code=1, reason="Error")))
    assert api.get(url).json() == {
        "failure": "agent bundle exited with code 1 (Error). Last log lines:\nFATAL: bad config",
        "not_ready": [],
    }


def test_bundle_is_reported_before_the_services(api: TestClient, k8s: FakeK8sClient) -> None:
    _bundle_pod(k8s, services=True)
    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").json() == {
        "failure": None,
        "not_ready": ["agent-bundle", "db"],
    }


def test_exec_never_reaches_the_bundle(api: TestClient, k8s: FakeK8sClient) -> None:
    _bundle_pod(k8s)
    response = api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/agent-bundle/exec", json={"command": ["id"]})
    assert response.status_code == 404


def test_sandbox_without_services_is_not_found(api: TestClient, k8s: FakeK8sClient) -> None:
    k8s.pods[0].metadata.annotations = {}
    response = api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "NEMO::NOT_FOUND"


class FakeStream:
    """A kubernetes exec stream: one stdout chunk, one stderr chunk, then closed.

    With ``open_forever`` it never closes, like a command that hangs, for the timeout path.
    """

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0, open_forever: bool = False) -> None:
        self._out = [stdout]
        self._err = [stderr] if stderr else []
        self._open_forever = open_forever
        self.returncode = returncode
        self.closed = False

    def is_open(self) -> bool:
        return self._open_forever or bool(self._out or self._err)

    def update(self, timeout: float) -> None:
        pass

    def peek_stdout(self) -> bool:
        return bool(self._out)

    def read_stdout(self) -> str:
        return self._out.pop()

    def peek_stderr(self) -> bool:
        return bool(self._err)

    def read_stderr(self) -> str:
        return self._err.pop()

    def close(self) -> None:
        self.closed = True


def test_exec_runs_in_the_service_container(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def stream(fn: Any, name: str, namespace: str, **kwargs: Any) -> FakeStream:
        calls.append({"name": name, "namespace": namespace, **kwargs})
        return FakeStream("PONG\n", "warn\n", returncode=3)

    monkeypatch.setattr(ext, "stream", stream)

    response = api.post(
        f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec", json={"command": ["redis-cli", "ping"]}
    )

    assert response.json() == {"exit_code": 3, "stdout": "PONG\n", "stderr": "warn\n"}
    assert calls[0]["name"] == "sbx-1-pod"
    assert calls[0]["container"] == "db"
    assert calls[0]["command"] == ["redis-cli", "ping"]


def _as_tenant(monkeypatch: pytest.MonkeyPatch, name: str, namespace: str) -> None:
    """Make the routes see a request from this tenant, as the server's API-key middleware would set it."""
    monkeypatch.setattr(ext, "get_current_tenant", lambda: types.SimpleNamespace(name=name, namespace=namespace))


def test_routes_search_the_provider_namespace_without_tenants(
    api: TestClient, k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "stream", lambda *args, **kwargs: FakeStream("ok"))
    api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services")
    api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec", json={"command": ["id"]})
    assert k8s.listed_namespaces == ["default", "default"]


def test_routes_search_the_callers_tenant_namespace(
    api: TestClient, k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _as_tenant(monkeypatch, "team-a", "sandbox-a")
    monkeypatch.setattr(ext, "stream", lambda *args, **kwargs: FakeStream("ok"))

    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").status_code == 200
    assert (
        api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec", json={"command": ["id"]}).status_code == 200
    )

    # The thread pool doesn't carry the request's context, so the namespace must reach it as an argument.
    assert k8s.listed_namespaces == ["sandbox-a", "sandbox-a"]


@pytest.mark.parametrize(("pod_label", "status"), [("team-a", 200), (None, 200), ("team-b", 404)])
def test_tenants_sharing_a_namespace_cant_see_each_others_sandboxes(
    api: TestClient, k8s: FakeK8sClient, monkeypatch: pytest.MonkeyPatch, pod_label: str | None, status: int
) -> None:
    _as_tenant(monkeypatch, "team-a", "shared")
    k8s.pods[0].metadata.labels = {"opensandbox.io/tenant": pod_label} if pod_label else {}
    monkeypatch.setattr(ext, "stream", lambda *args, **kwargs: FakeStream("ok"))

    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").status_code == status
    exec_url = f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec"
    assert api.post(exec_url, json={"command": ["id"]}).status_code == status


def test_a_tenants_label_is_ignored_without_tenants(api: TestClient, k8s: FakeK8sClient) -> None:
    k8s.pods[0].metadata.labels = {"opensandbox.io/tenant": "team-b"}
    assert api.get(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services").status_code == 200


@pytest.mark.parametrize("container", ["sandbox", "egress", "execd-installer", "cache"])
def test_exec_only_reaches_services(api: TestClient, container: str) -> None:
    response = api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/{container}/exec", json={"command": ["id"]})
    assert response.status_code == 404


def test_exec_is_bounded(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "_exec_slots", types.SimpleNamespace(locked=lambda: True))
    response = api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec", json={"command": ["id"]})
    assert response.status_code == 429


def test_exec_output_limit_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "EXEC_OUTPUT_LIMIT_BYTES", 4)
    with pytest.raises(ext.HTTPException) as too_large:
        ext._read_output(FakeStream("12345"), timeout=10)
    assert too_large.value.status_code == 413

    with pytest.raises(ext.HTTPException) as timed_out:
        ext._read_output(FakeStream("", open_forever=True), timeout=0)
    assert timed_out.value.status_code == 504


@pytest.mark.parametrize("body", [{}, {"command": []}, {"command": ["id"], "timeout": 0}, {"command": ["id"], "x": 1}])
def test_exec_rejects_bad_requests(api: TestClient, body: dict[str, Any]) -> None:
    assert api.post(f"/v1/nemo-ext/sandboxes/{SANDBOX_ID}/services/db/exec", json=body).status_code == 422


def test_install_registers_the_provider_before_the_server_builds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_factory, "_PROVIDER_REGISTRY", dict(provider_factory._PROVIDER_REGISTRY))
    monkeypatch.delenv("SANDBOX_CONFIG_PATH", raising=False)
    seen: dict[str, Any] = {}

    class FakeMain(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Stands in for the server's main module, which builds the provider and the app when imported."""

        def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
            return importlib.util.spec_from_loader(fullname, self) if fullname == "opensandbox_server.main" else None

        def create_module(self, spec: Any) -> None:
            return None

        def exec_module(self, module: types.ModuleType) -> None:
            seen["provider"] = provider_factory._PROVIDER_REGISTRY[provider_factory.PROVIDER_TYPE_BATCHSANDBOX]
            seen["config"] = os.environ.get("SANDBOX_CONFIG_PATH")
            module.app = FastAPI()  # ty: ignore[unresolved-attribute]

    monkeypatch.delitem(sys.modules, "opensandbox_server.main", raising=False)
    monkeypatch.delattr(opensandbox_server, "main", raising=False)
    monkeypatch.setattr(sys, "meta_path", [FakeMain(), *sys.meta_path])
    # Install replaces this; setting it here has monkeypatch put the stock one back afterwards.
    monkeypatch.setattr(
        kubernetes_service, "_attach_egress_auth_headers", kubernetes_service._attach_egress_auth_headers
    )
    monkeypatch.setattr(kubernetes_service, "_is_unschedulable_status", kubernetes_service._is_unschedulable_status)

    ext.install(["opensandbox-server", "--config", "/etc/opensandbox/config.toml"])

    assert seen == {"provider": ext.NemoComposeProvider, "config": "/etc/opensandbox/config.toml"}
    # The service's own token attachment is replaced, so the token follows a moved egress policy API.
    assert kubernetes_service._attach_egress_auth_headers is ext.attach_egress_auth_headers
    assert kubernetes_service._is_unschedulable_status is ext.end_create_wait
    app = sys.modules["opensandbox_server.main"].app
    # 503 rather than 404: the route is there, but this test built no provider.
    assert TestClient(app).get("/v1/nemo-ext/health").status_code == 503


def _run_loader(tmp_path: Path, program: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a stand-in ``program`` with the extension directory on PYTHONPATH, as the server image does."""
    script = tmp_path / program
    script.write_text("print('served')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(EXT_DIR), str(SRC_DIR)])}
    return subprocess.run([sys.executable, str(script), *args], env=env, capture_output=True, text=True, timeout=120)


def test_loader_ignores_other_programs(tmp_path: Path) -> None:
    result = _run_loader(tmp_path, "other-tool", "--reload")
    assert (result.returncode, result.stdout) == (0, "served\n")


def test_loader_stops_the_server_when_the_extension_cant_load(tmp_path: Path) -> None:
    result = _run_loader(tmp_path, "opensandbox-server", "--reload")
    assert result.returncode == 1
    assert "served" not in result.stdout
    assert "nemo-ext: not starting without the compose services extension" in result.stderr
    assert "--reload" in result.stderr


# A service on 18080, where the egress policy API listens, as the spec builder writes it.
SPEC_ON_18080 = SPEC.replace('"ports":[5432]', '"ports":[18080]').replace('"main":', '"egress_api_port":18081,"main":')


def test_a_service_on_18080_moves_the_egress_api(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    _create(provider, {POD_ANNOTATION: SPEC_ON_18080}, egress_auth_token="egress-token")

    manifest = k8s.created[0]
    egress = next(c for c in manifest["spec"]["template"]["spec"]["initContainers"] if c["name"] == "egress")
    assert {"name": "OPENSANDBOX_EGRESS_HTTP_ADDR", "value": ":18081"} in egress["env"]
    assert [p["containerPort"] for p in egress["ports"]] == [18081]
    # The server's own probe would otherwise reach the service and never see the sidecar ready.
    assert egress["readinessProbe"]["httpGet"]["port"] == 18081
    assert egress["startupProbe"]["httpGet"]["port"] == 18081
    assert manifest["metadata"]["annotations"][EGRESS_API_PORT_ANNOTATION] == "18081"


def test_a_stock_sandbox_keeps_the_egress_api_on_18080(provider: ext.NemoComposeProvider, k8s: FakeK8sClient) -> None:
    _create(provider, {POD_ANNOTATION: SPEC}, egress_auth_token="egress-token")

    manifest = k8s.created[0]
    egress = next(c for c in manifest["spec"]["template"]["spec"]["initContainers"] if c["name"] == "egress")
    assert all(e["name"] != "OPENSANDBOX_EGRESS_HTTP_ADDR" for e in egress["env"])
    assert egress["readinessProbe"]["httpGet"]["port"] == 18080
    assert EGRESS_API_PORT_ANNOTATION not in manifest["metadata"].get("annotations", {})


@pytest.fixture
def endpoint_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extension's endpoint auth, installed in the server's service layer as ``install`` does."""
    monkeypatch.setattr(kubernetes_service, "_attach_egress_auth_headers", ext.attach_egress_auth_headers)


def _get_endpoint(provider: ext.NemoComposeProvider, workload: dict[str, Any], port: int) -> Any:
    """What a client gets from the stock service's ``get_endpoint``, which attaches the auth headers."""
    provider.get_workload = lambda **_: workload  # ty: ignore[invalid-assignment]
    service = types.SimpleNamespace(workload_provider=provider, namespace="nemo", validate_port=lambda _: None)
    return KubernetesSandboxService.get_endpoint(service, SANDBOX_ID, port)  # ty: ignore[invalid-argument-type]


def _egress_workload(moved_port: int | None) -> dict[str, Any]:
    """A running sandbox with an egress token, whose egress policy API was moved to ``moved_port`` if given."""
    workload = _workload({POD_ANNOTATION: SPEC_ON_18080}, "Running", "10.0.0.5")
    workload["metadata"]["annotations"][SANDBOX_EGRESS_AUTH_TOKEN_METADATA_KEY] = "egress-token"
    if moved_port is not None:
        workload["metadata"]["annotations"][EGRESS_API_PORT_ANNOTATION] = str(moved_port)
    return workload


@pytest.mark.usefixtures("endpoint_auth")
@pytest.mark.parametrize(("port", "authorized"), [(18081, True), (18080, False), (5432, False)])
def test_the_egress_token_goes_only_to_the_moved_api(
    provider: ext.NemoComposeProvider, port: int, authorized: bool
) -> None:
    # Through the service, which attaches the token after the provider has built the endpoint: a client that
    # reuses the returned headers for a service on 18080 would otherwise hand it the sidecar's credential.
    endpoint = _get_endpoint(provider, _egress_workload(moved_port=18081), port)

    token_headers = build_egress_auth_headers("egress-token")
    assert (token_headers.items() <= (endpoint.headers or {}).items()) is authorized


@pytest.mark.usefixtures("endpoint_auth")
@pytest.mark.parametrize(("port", "authorized"), [(18080, True), (5432, False)])
def test_the_egress_token_stays_on_18080_when_the_api_was_not_moved(
    provider: ext.NemoComposeProvider, port: int, authorized: bool
) -> None:
    endpoint = _get_endpoint(provider, _egress_workload(moved_port=None), port)

    token_headers = build_egress_auth_headers("egress-token")
    assert (token_headers.items() <= (endpoint.headers or {}).items()) is authorized


def test_a_sandbox_without_an_egress_sidecar_records_no_moved_port(
    provider: ext.NemoComposeProvider, k8s: FakeK8sClient
) -> None:
    # Without a network policy the server adds no egress sidecar, so there's no API to move.
    _create(provider, {POD_ANNOTATION: SPEC_ON_18080}, network_policy=None)

    assert EGRESS_API_PORT_ANNOTATION not in k8s.created[0]["metadata"].get("annotations", {})
