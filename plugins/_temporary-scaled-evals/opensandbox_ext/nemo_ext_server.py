# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Compose services extension for the OpenSandbox server's Kubernetes BatchSandbox runtime.

A create request whose ``extensions`` carry ``EXTENSION_KEY`` (a JSON ``ComposeServices``) gets
those services as extra containers in its pod. One that carries ``AGENT_BUNDLE_EXTENSION_KEY`` (a
JSON ``SandboxAgentBundle``) gets an init container that installs the agent into ``/installed-agent``.
The stock server copies each extension onto the pod as an annotation (``POD_ANNOTATION``,
``AGENT_BUNDLE_POD_ANNOTATION``), so the spec travels with the manifest:

* ``NemoComposeProvider._merge_pod_spec_extras`` runs on the finished manifest, just before the
  stock code creates it, and adds the bundle (``nemo_ext_pod.add_agent_bundle``) and the services
  (``nemo_ext_pod.add_services``). An invalid spec raises ``ValueError``, which the stock server
  returns as a 400.
* The routes below read the same annotations from the pod, so they work for any sandbox and
  survive server restarts. They sit behind the server's API-key middleware, and with tenants
  configured they look in the caller's tenant namespace and don't see another tenant's sandboxes:

  * ``GET /v1/nemo-ext/health``: lets clients check the extension is loaded, and which features
    it has, because a stock server ignores unknown extensions and would create the sandbox
    without the services or the bundle.
  * ``GET /v1/nemo-ext/sandboxes/{id}/services``: whether the bundle or a service failed, and
    which aren't ready yet.
  * ``POST /v1/nemo-ext/sandboxes/{id}/services/{name}/exec``: runs a command in one service.

``NemoComposeProvider.get_status`` reports such a sandbox ``Allocated`` once its pod has an IP,
so create returns while the services start; the client waits for ``main`` (which Kubernetes starts
after the services) itself, and reads failures from the services route.

Loaded by ``sitecustomize.py``. Python 3.10, like the server image.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException
from kubernetes import client as k8s
from kubernetes.stream import stream
from nemo_ext_pod import add_agent_bundle, add_services, agent_bundle_state, move_egress_api, service_states
from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_CONTAINER,
    AGENT_BUNDLE_EXTENSION_KEY,
    AGENT_BUNDLE_POD_ANNOTATION,
    EGRESS_API_PORT_ANNOTATION,
    EXTENSION_FEATURES,
    EXTENSION_KEY,
    POD_ANNOTATION,
    ComposeServices,
    SandboxAgentBundle,
)
from opensandbox_server.config import CONFIG_ENV_VAR
from opensandbox_server.services.constants import SANDBOX_EGRESS_AUTH_TOKEN_METADATA_KEY, SANDBOX_ID_LABEL
from opensandbox_server.services.endpoint_auth import build_egress_auth_headers, merge_endpoint_headers
from opensandbox_server.services.k8s import provider_factory
from opensandbox_server.services.k8s.batchsandbox_provider import BatchSandboxProvider
from pydantic import BaseModel, ConfigDict, Field

try:
    from opensandbox_server.tenants.context import get_current_tenant
except ImportError:  # Servers before multi-tenancy (opensandbox-server 0.2.2) have no tenants.

    def get_current_tenant() -> Any:
        """No tenant: every sandbox is in the provider's namespace."""
        return None


# Under the server's logger tree, which is the only one its logging config enables.
logger = logging.getLogger("opensandbox_server.nemo_ext")

# The label a server with tenants puts on a tenant's sandboxes and their pods. Spelled out here because
# ``SANDBOX_TENANT_LABEL`` doesn't exist in servers without tenants, and importing it would stop them.
TENANT_LABEL = "opensandbox.io/tenant"

# Exec buffers a command's output in memory; this caps it per call.
EXEC_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024
# Concurrent execs; more get a 429. The pool has headroom for the services route.
EXEC_CONCURRENCY = 16
_pool = ThreadPoolExecutor(max_workers=EXEC_CONCURRENCY * 2, thread_name_prefix="nemo-ext")
_exec_slots = asyncio.Semaphore(EXEC_CONCURRENCY)
LOG_TAIL_LINES = 20

# The provider the server built; None if it runs another runtime.
_provider: NemoComposeProvider | None = None


class NemoComposeProvider(BatchSandboxProvider):
    """The stock BatchSandbox provider, plus the Compose services of a create request that carries them."""

    def __init__(self, k8s_client: Any, app_config: Any = None) -> None:
        """Build the stock provider and register it as the one the extension's routes use."""
        super().__init__(k8s_client, app_config=app_config)
        kube = app_config.kubernetes if app_config is not None else None
        self.namespace: str = (kube.namespace if kube is not None else None) or "default"
        global _provider
        _provider = self

    def _merge_pod_spec_extras(self, batchsandbox: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """The stock merge, then the services from the pod annotation, if the request carried them.

        The stock code calls this on the finished BatchSandbox manifest right before creating it,
        so the services are added to exactly the pod the server would otherwise create.
        """
        super()._merge_pod_spec_extras(batchsandbox, *args, **kwargs)
        template = batchsandbox["spec"]["template"]
        annotations = (template.get("metadata") or {}).get("annotations", {})
        pull_policy = self.image_pull_policy or "IfNotPresent"

        # pydantic's ValidationError is a ValueError, so the client gets a 400 with every problem listed.
        # The bundle goes first, so the agent is installed before any service starts.
        raw_bundle = annotations.get(AGENT_BUNDLE_POD_ANNOTATION)
        if raw_bundle is not None:
            add_agent_bundle(template["spec"], SandboxAgentBundle.model_validate_json(raw_bundle), pull_policy)

        raw = annotations.get(POD_ANNOTATION)
        if raw is not None:
            services = ComposeServices.model_validate_json(raw)
            add_services(template["spec"], services, pull_policy)

            # A service needs 18080, so move the egress policy API out of its way. A sandbox without a network policy
            # has no egress sidecar, so there's nothing to move or record.
            if services.egress_api_port is not None and move_egress_api(template["spec"], services.egress_api_port):
                # Record the moved port on the BatchSandbox, next to the egress token the server stores there, so
                # get_endpoint_info can find both later.
                workload_annotations = batchsandbox.setdefault("metadata", {}).setdefault("annotations", {})
                workload_annotations[EGRESS_API_PORT_ANNOTATION] = str(services.egress_api_port)

    def get_endpoint_info(self, workload: dict[str, Any], port: int, sandbox_id: str) -> Any:
        """The stock endpoint, plus the egress auth token when ``port`` is where we moved the egress policy API.

        The stock server only attaches the token for port 18080, so without this the egress API would reject our
        client's calls on the moved port. The server still attaches the token for 18080, where a compose service now
        listens, but our client never calls that port on these sandboxes.
        """
        endpoint = super().get_endpoint_info(workload, port, sandbox_id)
        if endpoint is None:
            return None

        annotations = (workload.get("metadata") or {}).get("annotations") or {}
        moved_egress_port = annotations.get(EGRESS_API_PORT_ANNOTATION)
        token = annotations.get(SANDBOX_EGRESS_AUTH_TOKEN_METADATA_KEY)
        if token and moved_egress_port == str(port):
            endpoint.headers = merge_endpoint_headers(endpoint.headers, build_egress_auth_headers(token))

        return endpoint

    def get_status(self, workload: dict[str, Any]) -> dict[str, Any]:
        """The stock status, except a sandbox with services or a bundle is ``Allocated`` once its pod has an IP.

        Kubernetes starts the sandbox container only after every init container and sidecar is up,
        and the stock create waits for that. Reporting the pod as allocated once it has an IP instead
        lets create return (the client looks up its endpoints next, so the IP must be known), and a
        failing or slow service or bundle is seen by the client's wait, not hidden behind the create call.
        """
        status = super().get_status(workload)
        if status["state"] != "Pending" or not self._parse_pod_ip(workload):
            return status

        template = (workload.get("spec") or {}).get("template") or {}
        annotations = (template.get("metadata") or {}).get("annotations") or {}
        if POD_ANNOTATION in annotations:
            return {
                **status,
                "state": "Allocated",
                "reason": "SERVICES_STARTING",
                "message": "Compose services are starting",
            }
        if AGENT_BUNDLE_POD_ANNOTATION in annotations:
            return {
                **status,
                "state": "Allocated",
                "reason": "AGENT_BUNDLE_INSTALLING",
                "message": "The agent bundle is installing",
            }
        return status

    def _create_workload_from_pool(self, *args: Any, **kwargs: Any) -> Any:
        """The stock pooled create, refused for a request with services.

        A pooled pod already exists, so there is no pod spec to add services to.
        """
        annotations = kwargs.get("annotations") or {}

        if POD_ANNOTATION in annotations:
            raise ValueError(f"extensions[{EXTENSION_KEY!r}] can't be combined with poolRef")
        if AGENT_BUNDLE_POD_ANNOTATION in annotations:
            raise ValueError(f"extensions[{AGENT_BUNDLE_EXTENSION_KEY!r}] can't be combined with poolRef")
        return super()._create_workload_from_pool(*args, **kwargs)

    def find_pod(
        self, sandbox_id: str, namespace: str, tenant: str | None = None
    ) -> tuple[Any, ComposeServices | None, SandboxAgentBundle | None]:
        """The sandbox's pod, its services and its agent bundle; 404 if it has no pod or neither.

        ``namespace`` is where the caller's sandboxes are, and ``tenant`` the caller's tenant name when the
        server has tenants (see ``_caller_scope``). Tenants can share a namespace, and the stock server hides
        one tenant's sandbox from the others as not found, so a pod labeled for another tenant is not found here too.
        """
        pods = self.k8s_client.list_pods(namespace=namespace, label_selector=f"{SANDBOX_ID_LABEL}={sandbox_id}")
        pod = pods[0] if pods else None
        if pod is not None and tenant is not None:
            owner = (pod.metadata.labels or {}).get(TENANT_LABEL)
            # An unlabeled pod isn't any tenant's, as in the stock server.
            if owner is not None and owner != tenant:
                pod = None
        annotations = (pod.metadata.annotations or {}) if pod is not None else {}
        raw_services = annotations.get(POD_ANNOTATION)
        raw_bundle = annotations.get(AGENT_BUNDLE_POD_ANNOTATION)

        # A sandbox the extension added nothing to isn't one these routes know about.
        if pod is None or (raw_services is None and raw_bundle is None):
            raise _error(404, "NOT_FOUND", f"sandbox {sandbox_id} has no pod with compose services or an agent bundle")

        return (
            pod,
            ComposeServices.model_validate_json(raw_services) if raw_services is not None else None,
            SandboxAgentBundle.model_validate_json(raw_bundle) if raw_bundle is not None else None,
        )

    def services_status(self, sandbox_id: str, namespace: str, tenant: str | None = None) -> dict[str, Any]:
        """``{"failure": str | None, "not_ready": [names]}`` for the sandbox's agent bundle and services.

        A failure names the first container that can't come up, in start order, and includes its
        last log lines, so the client can fail the trial with the cause instead of a timeout. The
        bundle's init container is listed in ``not_ready`` as ``agent-bundle`` until it exits 0.
        """
        pod, services, bundle = self.find_pod(sandbox_id, namespace, tenant)
        not_ready: list[str] = []

        # The bundle's init container runs before any service, so it's checked first.
        if bundle is not None:
            bundle_failure, installed = agent_bundle_state(pod)
            if bundle_failure is not None:
                return {
                    "failure": self._failure(pod, AGENT_BUNDLE_CONTAINER, "agent bundle", bundle_failure),
                    "not_ready": [],
                }
            if not installed:
                not_ready.append(AGENT_BUNDLE_CONTAINER)

        if services is not None:
            failure, services_not_ready = service_states(pod, services)
            not_ready.extend(services_not_ready)
            if failure is not None:
                name, why = failure
                return {"failure": self._failure(pod, name, f"service {name!r}", why), "not_ready": not_ready}
        return {"failure": None, "not_ready": not_ready}

    def _failure(self, pod: Any, container: str, label: str, reason: str) -> str:
        """``"<label> <reason>"`` for a container that can't come up, with its last log lines."""
        # A container that never started has no logs; report the failure anyway.
        try:
            tail = self.k8s_client.get_core_v1_api().read_namespaced_pod_log(
                pod.metadata.name, pod.metadata.namespace, container=container, tail_lines=LOG_TAIL_LINES
            )
        except Exception as exc:
            tail = f"<logs unavailable: {exc.__class__.__name__}>"
        return f"{label} {reason}. Last log lines:\n{tail}"

    def exec(
        self, sandbox_id: str, service: str, req: ExecRequest, namespace: str, tenant: str | None = None
    ) -> ExecResponse:
        """Run ``req.command`` in one of the sandbox's services through the Kubernetes exec API.

        Blocking; the route runs it on the extension's thread pool. 404 if ``service`` isn't one of
        the sandbox's services, 504 on timeout, 413 if the output passes the limit.
        """
        pod, services, _ = self.find_pod(sandbox_id, namespace, tenant)
        # Only the task's services, never sandbox, egress, execd or the agent bundle.
        if services is None or service not in services.names():
            raise _error(404, "NOT_FOUND", f"{service!r} is not a compose service of sandbox {sandbox_id}")

        # stream() patches the ApiClient it runs on, so each exec gets its own.
        with k8s.ApiClient() as api_client:
            ws = stream(
                k8s.CoreV1Api(api_client).connect_get_namespaced_pod_exec,
                pod.metadata.name,
                pod.metadata.namespace,
                container=service,
                command=req.command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            try:
                out, err = _read_output(ws, req.timeout)
                code = ws.returncode
            finally:
                ws.close()
        return ExecResponse(exit_code=code if code is not None else -1, stdout=out, stderr=err)


def _read_output(ws: Any, timeout: float) -> tuple[str, str]:
    """Read an exec stream's stdout and stderr until the command exits.

    Raises a 504 after ``timeout`` seconds and a 413 once the output passes
    ``EXEC_OUTPUT_LIMIT_BYTES``, since it's all held in memory.
    """
    out: list[str] = []
    err: list[str] = []
    size = 0
    deadline = time.monotonic() + timeout
    while ws.is_open():
        if time.monotonic() > deadline:
            raise _error(504, "EXEC_TIMEOUT", f"command timed out after {timeout}s")
        # Wait up to a second for frames, so the deadline is checked at least that often.
        ws.update(timeout=1)
        if ws.peek_stdout():
            out.append(ws.read_stdout())
            size += len(out[-1])
        if ws.peek_stderr():
            err.append(ws.read_stderr())
            size += len(err[-1])
        if size > EXEC_OUTPUT_LIMIT_BYTES:
            raise _error(413, "EXEC_OUTPUT_TOO_LARGE", f"command output exceeds {EXEC_OUTPUT_LIMIT_BYTES} bytes")
    return "".join(out), "".join(err)


def _error(status: int, code: str, message: str) -> HTTPException:
    """An HTTPException in the server's error shape."""
    return HTTPException(status_code=status, detail={"code": f"NEMO::{code}", "message": message})


def _require_provider() -> NemoComposeProvider:
    """The extension's provider; 503 if the server built another runtime's provider instead."""
    if _provider is None:
        raise _error(503, "EXTENSION_INACTIVE", "the server is not using the batchsandbox runtime")
    return _provider


def _caller_scope(provider: NemoComposeProvider) -> tuple[str, str | None]:
    """``(namespace, tenant name)`` for the request being served.

    A server with tenants resolves the caller's API key to a tenant and creates its sandboxes in that tenant's
    namespace. Without tenants every sandbox is in the provider's namespace, and there is no tenant name.
    Read this in the route, not in the thread pool, which doesn't carry the request's context.
    """
    tenant = get_current_tenant()
    if tenant is None:
        return provider.namespace, None
    return tenant.namespace, tenant.name


class ExecRequest(BaseModel):
    """The argv to run, without a shell, and how many seconds to wait for it."""

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    timeout: float = Field(default=300, gt=0, le=3600)


class ExecResponse(BaseModel):
    """``exit_code`` is -1 when Kubernetes reported none."""

    exit_code: int
    stdout: str
    stderr: str


router = APIRouter(prefix="/v1/nemo-ext")


@router.get("/health")
async def health() -> dict[str, Any]:
    """200 when the extension is loaded and its provider is active; clients check this before create."""
    _require_provider()
    return {"status": "ok", "extension": "nemo-compose-services", "features": list(EXTENSION_FEATURES)}


@router.get("/sandboxes/{sandbox_id}/services")
async def services_status(sandbox_id: str) -> dict[str, Any]:
    """Which of the sandbox's services failed or aren't ready; see ``NemoComposeProvider.services_status``."""
    provider = _require_provider()
    namespace, tenant = _caller_scope(provider)
    # The Kubernetes client blocks, so it runs off the event loop.
    return await asyncio.get_running_loop().run_in_executor(
        _pool, provider.services_status, sandbox_id, namespace, tenant
    )


@router.post("/sandboxes/{sandbox_id}/services/{service}/exec", response_model=ExecResponse)
async def exec_in_service(sandbox_id: str, service: str, req: ExecRequest) -> ExecResponse:
    """Run a command in one service; see ``NemoComposeProvider.exec``. 429 when all exec slots are busy."""
    provider = _require_provider()
    namespace, tenant = _caller_scope(provider)
    # Refuse rather than queue, so a burst of execs can't tie up the server's threads.
    if _exec_slots.locked():
        raise _error(429, "EXEC_BUSY", f"{EXEC_CONCURRENCY} execs are already running; retry later")
    async with _exec_slots:
        return await asyncio.get_running_loop().run_in_executor(
            _pool, provider.exec, sandbox_id, service, req, namespace, tenant
        )


def install(argv: list[str]) -> None:
    """Load the extension into the server process that ``argv`` starts. Raises if it can't."""
    # Overriding a method the server no longer has would silently drop the services.
    for name in (
        "_merge_pod_spec_extras",
        "_create_workload_from_pool",
        "get_status",
        "_parse_pod_ip",
        "get_endpoint_info",
    ):
        if not callable(getattr(BatchSandboxProvider, name, None)):
            raise RuntimeError(f"BatchSandboxProvider.{name} is missing; this server version is unsupported")
    if "--reload" in argv:
        raise RuntimeError("--reload starts the server in a subprocess, without the extension")

    # Importing the server's main module loads its config and builds the provider, so register
    # ours and point it at the config file first, as the server CLI would.
    if "--config" in argv:
        os.environ[CONFIG_ENV_VAR] = argv[argv.index("--config") + 1]
    provider_factory.register_provider(provider_factory.PROVIDER_TYPE_BATCHSANDBOX, NemoComposeProvider)
    import opensandbox_server.main as server_main

    server_main.app.include_router(router)
    logger.info("nemo-ext: compose services extension loaded (provider active: %s)", _provider is not None)
