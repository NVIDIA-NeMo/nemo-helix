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
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

import nemo_ext_server as ext
import opensandbox_server
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kubernetes.client import (
    V1ContainerState,
    V1ContainerStateTerminated,
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodStatus,
)
from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_EXTENSION_KEY,
    AGENT_BUNDLE_POD_ANNOTATION,
    EXTENSION_KEY,
    POD_ANNOTATION,
)
from opensandbox_server.api.schema import ImageSpec, NetworkPolicy, NetworkRule
from opensandbox_server.extensions import apply_extensions_to_annotations
from opensandbox_server.services.k8s import provider_factory
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


def _create(provider: BatchSandboxProvider, annotations: dict[str, str] | None, **kwargs: Any) -> None:
    """The stock create path for the test sandbox, with an egress policy."""
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
        network_policy=NetworkPolicy(defaultAction="deny", egress=[NetworkRule(action="allow", target="pypi.org")]),
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
    with pytest.raises(ValueError, match=r"ports \[18080\] are used"):
        _create(provider, {POD_ANNOTATION: SPEC.replace("5432", "18080")})
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
        "features": ["compose-services", "agent-bundle", "compose-main"],
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

    ext.install(["opensandbox-server", "--config", "/etc/opensandbox/config.toml"])

    assert seen == {"provider": ext.NemoComposeProvider, "config": "/etc/opensandbox/config.toml"}
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
