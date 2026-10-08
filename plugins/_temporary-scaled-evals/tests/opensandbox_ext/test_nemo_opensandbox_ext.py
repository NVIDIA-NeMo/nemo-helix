# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extension's hooks into the stock server: create, status, routes and loading."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import time
import tomllib
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nemo_opensandbox_ext as ext
import pytest
from _ext_doubles import (
    NAMESPACE,
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
from _server_doubles import SANDBOX_ID, FakeK8sClient, create
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kubernetes.client import V1ContainerStatus, V1Pod
from nemo_ext_pod import ROLES_ANNOTATION
from nemo_ext_spec import SERVICES_KEY, ServicesSpecError
from opensandbox_server.api.schema import ImageSpec
from opensandbox_server.services.k8s import provider_factory

# ---- create ------------------------------------------------------------------------------


def test_services_are_added_to_the_manifest_the_stock_server_builds(
    provider: ext.NemoServicesProvider, k8s: FakeK8sClient
) -> None:
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})

    template = k8s.created[0]["spec"]["template"]
    pod_spec = template["spec"]

    # The pod builder's own tests cover the details; this checks it ran on the real manifest.
    assert [c["name"] for c in pod_spec["initContainers"]] == ["execd-installer", "egress", "db", "migrate", "proxy"]
    assert [c["name"] for c in pod_spec["containers"]] == ["sandbox", "worker"]
    assert by_name(pod_spec["initContainers"])["db"]["imagePullPolicy"] == provider.image_pull_policy
    assert json.loads(template["metadata"]["annotations"][ROLES_ANNOTATION]) == SPEC_ROLES


@pytest.mark.parametrize("egress", [True, False])
def test_the_pod_builder_tests_use_the_stock_manifest_shape(
    provider: ext.NemoServicesProvider, k8s: FakeK8sClient, egress: bool
) -> None:
    create(provider, None, egress=egress)
    real = k8s.created[0]["spec"]["template"]["spec"]
    double = stock_batchsandbox(egress=egress)["spec"]["template"]["spec"]

    for key in ("initContainers", "containers", "volumes"):
        assert [c["name"] for c in real[key]] == [c["name"] for c in double[key]]
    if egress:
        real_egress, double_egress = by_name(real["containers"])["egress"], by_name(double["containers"])["egress"]
        assert real_egress["ports"] == double_egress["ports"]
        assert real_egress["readinessProbe"]["httpGet"] == double_egress["readinessProbe"]["httpGet"]


def test_requests_without_services_are_untouched(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    stock = FakeK8sClient()
    request: dict[str, Any] = {
        "sandbox_id": SANDBOX_ID,
        "namespace": NAMESPACE,
        "image_spec": ImageSpec(uri="registry.example.com/task/main:1"),
        "entrypoint": ["sleep", "infinity"],
        "env": {},
        "resource_limits": {},
        "labels": {"opensandbox.io/id": SANDBOX_ID},
        "expires_at": None,
        "execd_image": "registry.example.com/opensandbox/execd:v1",
    }
    provider_factory.BatchSandboxProvider(stock).create_workload(**request)  # ty: ignore[invalid-argument-type]
    provider.create_workload(**request)
    assert k8s.created == stock.created
    assert provider._tracked(SANDBOX_ID) is None


def test_invalid_spec_creates_nothing(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    with pytest.raises(ServicesSpecError):
        create(provider, {SERVICES_KEY: json.dumps(spec_dict(version=2))})
    assert k8s.created == []
    assert provider._tracked(SANDBOX_ID) is None


def test_pool_mode_is_rejected(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    with pytest.raises(ServicesSpecError, match="poolRef"):
        create(provider, {SERVICES_KEY: json.dumps(spec_dict()), "poolRef": "pool-a"})
    assert k8s.created == []


# ---- readiness and fail-fast -------------------------------------------------------------


def workload(*, ready: int = 0, pod_ip: bool = True) -> dict[str, Any]:
    """The BatchSandbox object the stock ``get_status`` reads; ``pod_ip`` makes it ``Allocated``."""
    status: dict[str, Any] = {"replicas": 1, "ready": ready, "allocated": 1}
    workload = {
        "metadata": {"name": SANDBOX_ID, "namespace": NAMESPACE, "labels": {"opensandbox.io/id": SANDBOX_ID}},
        "status": status,
    }
    if pod_ip:
        workload["metadata"]["annotations"] = {"sandbox.opensandbox.io/endpoints": '["10.0.0.5"]'}
    return workload


def test_stock_status_while_services_start_is_allocated(provider: ext.NemoServicesProvider) -> None:
    # The stock wait loop accepts "Allocated" (pod has an IP), which happens before any
    # init container runs; this is why the extension must hold the status.
    assert provider_factory.BatchSandboxProvider.get_status(provider, workload())["state"] == "Allocated"


def test_status_is_pending_until_every_service_is_ready(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})

    k8s.pods = [pod([running("egress"), running("db", ready=False)], [])]
    status = provider.get_status(workload())
    assert status["state"] == "Pending"
    assert status["message"] == "waiting for services: db, migrate, proxy, worker, main"

    k8s.pods = [
        pod(
            [running("egress"), running("db"), terminated("migrate", 0), running("proxy")],
            [running("sandbox"), running("worker", ready=False)],
        )
    ]
    assert provider.get_status(workload())["message"] == "waiting for services: worker"

    k8s.pods = [all_ready()]
    assert provider.get_status(workload())["state"] == "Allocated"
    # Once ready, the sandbox is no longer watched and status is the stock one.
    calls = k8s.pod_list_calls
    provider.get_status(workload())
    assert k8s.pod_list_calls == calls


@pytest.mark.parametrize(
    ("init", "containers", "phase", "failed"),
    [
        ([running("egress"), running("db"), terminated("migrate", 3)], [], "Failed", "migrate"),
        ([running("egress"), terminated("db", 1, restarts=3)], [], "Pending", "db"),
        ([running("egress"), waiting("db", "ImagePullBackOff")], [], "Pending", "db"),
        (
            [running("egress"), running("db"), terminated("migrate", 0), running("proxy")],
            [running("sandbox"), terminated("worker", 0)],
            "Running",
            "worker",
        ),
    ],
)
def test_failed_service_fails_the_create(
    provider: ext.NemoServicesProvider,
    k8s: FakeK8sClient,
    init: list[V1ContainerStatus],
    containers: list[V1ContainerStatus],
    phase: str,
    failed: str,
) -> None:
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})
    k8s.pods = [pod(init, containers, phase)]
    for _ in range(2):  # sticky: a client GET seeing it first doesn't hide it from the wait loop
        with pytest.raises(HTTPException) as info:
            provider.get_status(workload())
        assert info.value.status_code == 422
        detail: Any = info.value.detail
        assert detail["code"] == "NEMO::SERVICE_FAILED"
        assert f"service {failed!r} failed" in detail["message"]
        assert "boom: config missing" in detail["message"]


def test_a_sidecar_restart_below_the_threshold_is_not_a_failure(
    provider: ext.NemoServicesProvider, k8s: FakeK8sClient
) -> None:
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})
    k8s.pods = [pod([running("egress"), terminated("db", 1, restarts=1)], [])]
    assert provider.get_status(workload())["state"] == "Pending"


def test_untracked_sandboxes_never_raise(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    k8s.pods = [pod([terminated("db", 1, restarts=5)], [], "Failed")]
    assert provider.get_status(workload())["state"] == "Allocated"
    assert k8s.pod_list_calls == 0


def test_watch_expires(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    provider._watch_sec = -1
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})
    assert provider.get_status(workload())["state"] == "Allocated"


def test_pod_read_errors_keep_waiting(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> None:
    create(provider, {SERVICES_KEY: json.dumps(spec_dict())})

    def broken(**_: Any) -> list[V1Pod]:
        raise RuntimeError("apiserver unavailable")

    k8s.list_pods = broken  # ty: ignore[invalid-assignment]
    assert provider.get_status(workload())["state"] == "Pending"


# ---- routes ------------------------------------------------------------------------------


class FakeWs:
    """A kubernetes exec stream that yields the given output chunks, then closes with ``returncode``."""

    def __init__(self, stdout: list[str], stderr: list[str], returncode: int) -> None:
        self._stdout, self._stderr, self.returncode = stdout, stderr, returncode
        self.closed = False

    def is_open(self) -> bool:
        return bool(self._stdout or self._stderr)

    def update(self, timeout: float) -> None:
        pass

    def peek_stdout(self) -> bool:
        return bool(self._stdout)

    def read_stdout(self) -> str:
        return self._stdout.pop(0)

    def peek_stderr(self) -> bool:
        return bool(self._stderr)

    def read_stderr(self) -> str:
        return self._stderr.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def api(provider: ext.NemoServicesProvider, k8s: FakeK8sClient) -> TestClient:
    """An app with only the extension's routes, over a sandbox whose services are all up."""
    app = FastAPI()
    ext.attach_routes(app)
    k8s.pods = [all_ready()]
    return TestClient(app)


def test_health(api: TestClient) -> None:
    assert api.get("/v1/nemo-ext/health").json() == {"status": "ok", "extension": "nemo-services"}


def test_health_without_the_provider(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "_active_provider", None)
    resp = api.get("/v1/nemo-ext/health")
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "NEMO::EXTENSION_INACTIVE"


def test_exec_runs_in_the_service_container(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    ws = FakeWs(["hel", "lo"], ["warn"], 7)

    def fake_stream(method: Any, name: str, namespace: str, **kwargs: Any) -> FakeWs:
        calls.append({"name": name, "namespace": namespace, **kwargs})
        return ws

    monkeypatch.setattr(ext, "stream", fake_stream)
    resp = api.post(f"/v1/sandboxes/{SANDBOX_ID}/containers/db/exec", json={"command": ["sh", "-c", "echo hello"]})
    assert resp.status_code == 200
    assert resp.json() == {"exit_code": 7, "stdout": "hello", "stderr": "warn"}
    assert calls[0]["name"] == "sbx-1-pod" and calls[0]["namespace"] == NAMESPACE
    assert calls[0]["container"] == "db" and calls[0]["command"] == ["sh", "-c", "echo hello"]
    assert ws.closed


@pytest.mark.parametrize("container", ["sandbox", "egress", "execd-installer", "nope"])
def test_exec_refuses_non_service_containers(api: TestClient, container: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "stream", lambda *a, **k: pytest.fail("must not exec"))
    resp = api.post(f"/v1/sandboxes/{SANDBOX_ID}/containers/{container}/exec", json={"command": ["id"]})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "NEMO::NOT_FOUND"


def test_exec_for_unknown_sandbox(api: TestClient, k8s: FakeK8sClient) -> None:
    k8s.pods = []
    resp = api.post(f"/v1/sandboxes/{SANDBOX_ID}/containers/db/exec", json={"command": ["id"]})
    assert resp.status_code == 404


def test_exec_output_reader_times_out() -> None:
    class HungWs(FakeWs):
        def is_open(self) -> bool:
            return True

        def update(self, timeout: float) -> None:
            time.sleep(0.02)

    with pytest.raises(HTTPException) as raised:
        ext._read_exec_output(HungWs([], [], 0), timeout=0.01)
    assert raised.value.status_code == 504
    assert raised.value.detail == {"code": "NEMO::EXEC_TIMEOUT", "message": "command timed out after 0.01s"}


def test_exec_output_limit(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ext, "EXEC_OUTPUT_LIMIT_BYTES", 4)
    ws = FakeWs(["12345"], [], 0)
    monkeypatch.setattr(ext, "stream", lambda *a, **k: ws)
    resp = api.post(f"/v1/sandboxes/{SANDBOX_ID}/containers/db/exec", json={"command": ["cat", "big"]})
    assert resp.status_code == 413
    assert ws.closed


@pytest.mark.parametrize("body", [{"command": []}, {"command": ["id"], "timeout": 0}, {"command": ["id"], "x": 1}])
def test_exec_rejects_bad_requests(api: TestClient, body: dict[str, Any]) -> None:
    assert api.post(f"/v1/sandboxes/{SANDBOX_ID}/containers/db/exec", json=body).status_code == 422


# ---- loader ------------------------------------------------------------------------------

EXT_DIR = Path(ext.__file__).parent


@pytest.fixture
def sitecustomize() -> Iterator[types.ModuleType]:
    """Import ``sitecustomize.py`` under another name, removing its import hook afterwards."""
    before = list(sys.meta_path)
    spec = importlib.util.spec_from_file_location("nemo_ext_sitecustomize_under_test", EXT_DIR / "sitecustomize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.meta_path[:] = before


def test_loader_only_wraps_the_server_main_module(sitecustomize: types.ModuleType) -> None:
    import opensandbox_server

    finder = sitecustomize._Finder()
    package_path = list(opensandbox_server.__path__)
    assert finder.find_spec("opensandbox_server.config", package_path) is None
    target = finder.find_spec(sitecustomize.TARGET_MODULE, package_path)
    assert target is not None and isinstance(target.loader, sitecustomize._Loader)
    assert target.origin is not None and target.origin.endswith("opensandbox_server/main.py")


class FakeMain(importlib.machinery.SourceFileLoader):
    """Stands in for the server's main module: records the batchsandbox provider it sees, then builds an app."""

    def __init__(self) -> None:
        self.provider_at_main: Any = None

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        self.provider_at_main = provider_factory._PROVIDER_REGISTRY["batchsandbox"]
        module.app = FastAPI()  # ty: ignore[unresolved-attribute]


def test_loader_registers_before_main_runs_and_adds_routes_after(
    sitecustomize: types.ModuleType, provider: ext.NemoServicesProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_factory, "_PROVIDER_REGISTRY", dict(provider_factory._PROVIDER_REGISTRY))
    main = FakeMain()

    module = types.ModuleType("opensandbox_server.main")
    sitecustomize._Loader(main).exec_module(module)
    assert main.provider_at_main is ext.NemoServicesProvider
    assert TestClient(module.app).get("/v1/nemo-ext/health").status_code == 200


@pytest.mark.parametrize("version", ["0.2.3", None])
def test_loader_leaves_an_untested_server_unchanged(
    sitecustomize: types.ModuleType, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    monkeypatch.setattr(provider_factory, "_PROVIDER_REGISTRY", dict(provider_factory._PROVIDER_REGISTRY))
    stock = provider_factory._PROVIDER_REGISTRY["batchsandbox"]
    monkeypatch.setattr(sitecustomize, "server_version", lambda: version)
    main = FakeMain()

    module = types.ModuleType("opensandbox_server.main")
    sitecustomize._Loader(main).exec_module(module)
    assert main.provider_at_main is stock
    assert TestClient(module.app).get("/v1/nemo-ext/health").status_code == 404


def test_pinned_server_version_is_a_tested_one(sitecustomize: types.ModuleType) -> None:
    pyproject = tomllib.loads((EXT_DIR.parents[2] / "pyproject.toml").read_text())
    pins = [p for p in pyproject["dependency-groups"]["scaled-evals"] if p.startswith("opensandbox-server==")]
    assert len(pins) == 1
    assert pins[0].removeprefix("opensandbox-server==") in sitecustomize.TESTED_SERVER_VERSIONS
    assert sitecustomize.server_version() in sitecustomize.TESTED_SERVER_VERSIONS


# ---- deployment manifest -----------------------------------------------------------------


def test_committed_configmap_matches_the_sources() -> None:
    import render_configmap

    assert render_configmap.OUTPUT.read_text() == render_configmap.render(), (
        "run: uv run python plugins/_temporary-scaled-evals/opensandbox_ext/render_configmap.py"
    )
