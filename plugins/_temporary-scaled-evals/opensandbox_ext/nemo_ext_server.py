# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Compose services extension for the OpenSandbox server's Kubernetes BatchSandbox runtime.

A create request whose ``extensions`` carry ``EXTENSION_KEY`` (a JSON ``ComposeServices``) gets
those services as extra containers in its pod. The stock server copies that extension onto the
pod as the ``POD_ANNOTATION`` annotation, so the spec travels with the manifest:

* ``NemoComposeProvider._merge_pod_spec_extras`` runs on the finished manifest, just before the
  stock code creates it, and adds the services (``nemo_ext_pod.add_services``). An invalid spec
  raises ``ValueError``, which the stock server returns as a 400.
* The routes below read the same annotation from the pod, so they work for any sandbox and
  survive server restarts. They sit behind the server's API-key middleware:

  * ``GET /v1/nemo-ext/health``: lets clients check the extension is loaded, because a stock
    server ignores unknown extensions and would create the sandbox without the services.
  * ``GET /v1/nemo-ext/sandboxes/{id}/services``: which services failed or aren't ready yet.
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
from nemo_ext_pod import add_services, service_states
from opensandbox_server.config import CONFIG_ENV_VAR
from opensandbox_server.services.constants import SANDBOX_ID_LABEL
from opensandbox_server.services.k8s import provider_factory
from opensandbox_server.services.k8s.batchsandbox_provider import BatchSandboxProvider
from pydantic import BaseModel, ConfigDict, Field
from scaled_evals.harbor_opensandbox_services import EXTENSION_KEY, POD_ANNOTATION, ComposeServices

# Under the server's logger tree, which is the only one its logging config enables.
logger = logging.getLogger("opensandbox_server.nemo_ext")

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
        raw = (template.get("metadata") or {}).get("annotations", {}).get(POD_ANNOTATION)
        if raw is not None:
            # pydantic's ValidationError is a ValueError, so the client gets a 400 with every problem listed.
            services = ComposeServices.model_validate_json(raw)
            add_services(template["spec"], services, self.image_pull_policy or "IfNotPresent")

    def get_status(self, workload: dict[str, Any]) -> dict[str, Any]:
        """The stock status, except a sandbox with services counts as ``Allocated`` once its pod has an IP.

        Kubernetes starts the sandbox container only after every sidecar is up, and the stock
        create waits for that. Reporting the pod as allocated once it has an IP instead lets
        create return (the client looks up its endpoints next, so the IP must be known), and a
        failing or slow service is seen by the client's wait, not hidden behind the create call.
        """
        status = super().get_status(workload)
        template = (workload.get("spec") or {}).get("template") or {}
        if (
            status["state"] == "Pending"
            and self._parse_pod_ip(workload)
            and POD_ANNOTATION in ((template.get("metadata") or {}).get("annotations") or {})
        ):
            return {
                **status,
                "state": "Allocated",
                "reason": "SERVICES_STARTING",
                "message": "Compose services are starting",
            }
        return status

    def _create_workload_from_pool(self, *args: Any, **kwargs: Any) -> Any:
        """The stock pooled create, refused for a request with services.

        A pooled pod already exists, so there is no pod spec to add services to.
        """
        if POD_ANNOTATION in (kwargs.get("annotations") or {}):
            raise ValueError(f"extensions[{EXTENSION_KEY!r}] can't be combined with poolRef")
        return super()._create_workload_from_pool(*args, **kwargs)

    def find_pod(self, sandbox_id: str) -> tuple[Any, ComposeServices]:
        """The sandbox's pod and its services; 404 if it has no pod or no services."""
        pods = self.k8s_client.list_pods(namespace=self.namespace, label_selector=f"{SANDBOX_ID_LABEL}={sandbox_id}")
        raw = (pods[0].metadata.annotations or {}).get(POD_ANNOTATION) if pods else None
        if raw is None:
            raise _error(404, "NOT_FOUND", f"sandbox {sandbox_id} has no pod with compose services")
        return pods[0], ComposeServices.model_validate_json(raw)

    def services_status(self, sandbox_id: str) -> dict[str, Any]:
        """``{"failure": str | None, "not_ready": [names]}`` for the sandbox's services.

        A failure names the first service that can't come up and includes its last log lines,
        so the client can fail the trial with the cause instead of a timeout.
        """
        pod, services = self.find_pod(sandbox_id)
        failure, not_ready = service_states(pod, services)
        if failure is None:
            return {"failure": None, "not_ready": not_ready}
        name, why = failure
        # A container that never started has no logs; report the failure anyway.
        try:
            tail = self.k8s_client.get_core_v1_api().read_namespaced_pod_log(
                pod.metadata.name, pod.metadata.namespace, container=name, tail_lines=LOG_TAIL_LINES
            )
        except Exception as exc:
            tail = f"<logs unavailable: {exc.__class__.__name__}>"
        return {"failure": f"service {name!r} {why}. Last log lines:\n{tail}", "not_ready": not_ready}

    def exec(self, sandbox_id: str, service: str, req: ExecRequest) -> ExecResponse:
        """Run ``req.command`` in one of the sandbox's services through the Kubernetes exec API.

        Blocking; the route runs it on the extension's thread pool. 404 if ``service`` isn't one of
        the sandbox's services, 504 on timeout, 413 if the output passes the limit.
        """
        pod, services = self.find_pod(sandbox_id)
        # Only the task's services, never sandbox, egress or execd.
        if service not in services.names():
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
async def health() -> dict[str, str]:
    """200 when the extension is loaded and its provider is active; clients check this before create."""
    _require_provider()
    return {"status": "ok", "extension": "nemo-compose-services"}


@router.get("/sandboxes/{sandbox_id}/services")
async def services_status(sandbox_id: str) -> dict[str, Any]:
    """Which of the sandbox's services failed or aren't ready; see ``NemoComposeProvider.services_status``."""
    provider = _require_provider()
    # The Kubernetes client blocks, so it runs off the event loop.
    return await asyncio.get_running_loop().run_in_executor(_pool, provider.services_status, sandbox_id)


@router.post("/sandboxes/{sandbox_id}/services/{service}/exec", response_model=ExecResponse)
async def exec_in_service(sandbox_id: str, service: str, req: ExecRequest) -> ExecResponse:
    """Run a command in one service; see ``NemoComposeProvider.exec``. 429 when all exec slots are busy."""
    provider = _require_provider()
    # Refuse rather than queue, so a burst of execs can't tie up the server's threads.
    if _exec_slots.locked():
        raise _error(429, "EXEC_BUSY", f"{EXEC_CONCURRENCY} execs are already running; retry later")
    async with _exec_slots:
        return await asyncio.get_running_loop().run_in_executor(_pool, provider.exec, sandbox_id, service, req)


def install(argv: list[str]) -> None:
    """Load the extension into the server process that ``argv`` starts. Raises if it can't."""
    # Overriding a method the server no longer has would silently drop the services.
    for name in ("_merge_pod_spec_extras", "_create_workload_from_pool", "get_status", "_parse_pod_ip"):
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
