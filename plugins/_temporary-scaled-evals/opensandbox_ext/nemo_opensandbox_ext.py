# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo services extension for the OpenSandbox server's Kubernetes BatchSandbox runtime.

A sandbox create request that carries ``extensions["nemo.nvidia.com/services"]`` (a JSON
``ServicesSpec``, see ``nemo_ext_spec``) gets the listed services as extra containers in its
pod, the way the task's Docker Compose file would have run them next to ``main`` (see
``nemo_ext_pod``). This module wires that into the stock server, in three places:

* Create. ``NemoServicesProvider`` replaces the BatchSandbox provider. It lets the stock code
  build the BatchSandbox manifest as usual, and adds the services just before the manifest
  is sent to Kubernetes.
* Status. The stock server answers the create call once its status check reports the sandbox
  up, which happens as soon as the pod has an IP, before any service has started. While a
  sandbox with services is being created, ``get_status`` reports it ``Pending`` until every
  service is ready, and turns a service that can't start into a 422 ``NEMO::SERVICE_FAILED``
  with its last log lines. The stock create path then deletes the sandbox.
* Routes. ``POST /v1/sandboxes/{id}/containers/{name}/exec`` runs a command in one service
  container (Harbor runs some task steps inside services). It only reaches containers this
  extension added. ``GET /v1/nemo-ext/health`` lets clients check the extension is active,
  because a stock server would silently ignore the services. Both routes sit behind the
  server's API-key middleware.

Loaded by ``sitecustomize.py`` in this directory; only depends on ``opensandbox-server`` and
its dependencies. Everything that touches stock server internals lives in this module.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from kubernetes import client as k8s
from kubernetes.stream import stream
from nemo_ext_pod import inject_services, pod_roles, service_failure, services_waiting
from nemo_ext_spec import SERVICES_KEY, Role, ServicesSpec, ServicesSpecError, parse_spec
from opensandbox_server.services.constants import SANDBOX_ID_LABEL
from opensandbox_server.services.k8s import provider_factory
from opensandbox_server.services.k8s.batchsandbox_provider import BatchSandboxProvider
from opensandbox_server.services.k8s.client import K8sClient
from pydantic import BaseModel, ConfigDict, Field

# Under the server's logger tree, which is the only one its logging config enables.
logger = logging.getLogger("opensandbox_server.nemo_ext")

# The stock server polls a new sandbox's status every second by default, and clients may read it
# too. Reading the pod on every poll would load the Kubernetes API for no gain, so the pod is read
# at most this often.
STATUS_CHECK_INTERVAL_SEC = 2.0

# How many log lines of a failed service go into the 422 message.
LOG_TAIL_LINES = 20

# The exec route buffers a command's output in memory; this caps it per call.
EXEC_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024

# How much longer than the server's create timeout a new sandbox stays tracked; see _Creating.
WATCH_GRACE_SEC = 60.0


# ---- workload provider -------------------------------------------------------------------

# The services spec of the create call running in the current context. The stock create path
# builds the manifest and sends it with create_custom_object, without passing the request's
# extensions down, so this carries the spec from create_workload to _InjectingClient.
_creating_spec: contextvars.ContextVar[ServicesSpec | None] = contextvars.ContextVar("nemo_services_spec", default=None)

# Signatures of the stock methods this module wraps. Arguments are bound by name, so it
# doesn't matter whether the stock code passes them positionally or as keywords.
_CREATE_SIGNATURE = inspect.signature(BatchSandboxProvider.create_workload)
_CREATE_CR_SIGNATURE = inspect.signature(K8sClient.create_custom_object)

# The provider the server built. The HTTP routes use it to find a sandbox's pod; it stays None
# when the server runs another runtime (e.g. Docker), and the routes then return 503.
_active_provider: NemoServicesProvider | None = None


class _InjectingClient:
    """The ``K8sClient`` the provider's stock create path talks to: adds the services to the BatchSandbox it creates.

    Every other call passes through to the real client unchanged.
    """

    def __init__(self, inner: K8sClient, image_pull_policy: str) -> None:
        """Wrap ``inner``. Service containers get ``image_pull_policy``, like the stock sandbox container."""
        self._inner = inner
        self._image_pull_policy = image_pull_policy

    def __getattr__(self, name: str) -> Any:
        """Pass every other ``K8sClient`` call through unchanged."""
        return getattr(self._inner, name)

    def create_custom_object(self, *args: Any, **kwargs: Any) -> Any:
        """Add the current create call's services to the manifest if it is a BatchSandbox, then create it."""
        spec = _creating_spec.get()

        # Only a create call with services sets a spec, and only its BatchSandbox gets them;
        # any other custom object the stock code creates passes through untouched.
        if spec is not None:
            call = _CREATE_CR_SIGNATURE.bind(self._inner, *args, **kwargs).arguments
            if call["plural"] == "batchsandboxes":
                inject_services(call["body"], spec, self._image_pull_policy)

        return self._inner.create_custom_object(*args, **kwargs)


@dataclass
class _Creating:
    """Status-check state for one sandbox whose create call hasn't finished yet.

    Kept until the services are up, the create fails, or ``deadline`` passes, whichever comes
    first. The deadline is a little past the server's own create timeout, so a sandbox whose
    create call was abandoned is eventually forgotten.
    """

    roles: dict[str, Role]
    deadline: float
    next_check: float = 0.0
    message: str = "waiting for services"
    # Kept until the deadline so every status read (the create wait loop, or a client
    # GET that happened to look first) reports the same failure.
    failure: str | None = None
    # Status reads can run on several server threads at once; this serializes them per sandbox.
    lock: threading.Lock = field(default_factory=threading.Lock)


class NemoServicesProvider(BatchSandboxProvider):
    """The stock BatchSandbox provider, extended to add Compose-style services to the sandbox pod.

    Sandboxes created without a services spec behave exactly as with the stock provider.
    """

    def __init__(self, k8s_client: K8sClient, app_config: Any = None) -> None:
        """Build the stock provider, then route its manifest creation through ``_InjectingClient``."""
        super().__init__(k8s_client, app_config=app_config)
        kube = app_config.kubernetes if app_config is not None else None

        # The stock code only uses self.k8s_client, so swapping it intercepts its creates.
        # This module keeps the real client for its own pod reads.
        self.raw_k8s_client = k8s_client
        self.k8s_client = _InjectingClient(k8s_client, self.image_pull_policy or "IfNotPresent")
        self.namespace: str = (kube.namespace if kube is not None else None) or "default"

        # Track a sandbox a little longer than the server waits for it to start.
        self._watch_sec = float(kube.sandbox_create_timeout_seconds if kube is not None else 900) + WATCH_GRACE_SEC
        self._creating: dict[str, _Creating] = {}
        self._creating_lock = threading.Lock()

        global _active_provider
        _active_provider = self

    def create_workload(self, *args: Any, **kwargs: Any) -> Any:
        """Create the sandbox; with a services spec, add the services and track the sandbox until they are up.

        Raises ``ServicesSpecError`` (an HTTP 400) for an invalid spec, before anything is created.
        """
        call = _CREATE_SIGNATURE.bind(self, *args, **kwargs).arguments
        extensions = call.get("extensions") or {}
        raw = extensions.get(SERVICES_KEY)

        # No services: the stock create, untouched.
        if raw is None:
            return super().create_workload(*args, **kwargs)

        # A pooled sandbox gets a pod the pool created in advance, so there is no pod template to extend.
        if extensions.get("poolRef"):
            raise ServicesSpecError(f"extensions[{SERVICES_KEY!r}] is not supported together with poolRef")

        spec = parse_spec(raw)

        # Start tracking before the create, so the server's first status poll already sees it.
        sandbox_id = call["sandbox_id"]
        with self._creating_lock:
            self._creating[sandbox_id] = _Creating(spec.roles(), time.monotonic() + self._watch_sec)

        # Make the spec visible to _InjectingClient for the length of the stock create call.
        token = _creating_spec.set(spec)
        try:
            return super().create_workload(*args, **kwargs)
        except BaseException:
            # Nothing was created, so there is nothing to track.
            self._forget(sandbox_id)
            raise
        finally:
            _creating_spec.reset(token)

    def get_status(self, workload: dict[str, Any]) -> dict[str, Any]:
        """The sandbox's status, held at ``Pending`` while its services start.

        For a sandbox this provider is tracking, returns ``Pending`` (with the services still
        starting in the message) until every service is ready and the stock status also says
        the sandbox is up. Raises a 422 ``NEMO::SERVICE_FAILED`` once a service has failed.
        Every other sandbox gets the stock status.
        """
        status = super().get_status(workload)

        metadata = workload.get("metadata") or {}
        sandbox_id = (metadata.get("labels") or {}).get(SANDBOX_ID_LABEL) or metadata.get("name")
        if sandbox_id is None:
            return status

        entry = self._tracked(sandbox_id)
        if entry is None:
            return status

        with entry.lock:
            # Read the pod at most every STATUS_CHECK_INTERVAL_SEC; between reads, repeat the last answer.
            if entry.failure is None and time.monotonic() >= entry.next_check:
                entry.next_check = time.monotonic() + STATUS_CHECK_INTERVAL_SEC
                waiting = self._check(sandbox_id, metadata.get("namespace") or self.namespace, entry)

                # Every service is up. Once the stock status also says the sandbox is up, stop
                # tracking and hand it back to the stock status for good.
                if not waiting and entry.failure is None and status.get("state") in ("Running", "Allocated"):
                    self._forget(sandbox_id)
                    return status

                if waiting:
                    entry.message = f"waiting for services: {', '.join(waiting)}"

            if entry.failure is not None:
                raise HTTPException(status_code=422, detail={"code": "NEMO::SERVICE_FAILED", "message": entry.failure})

        # Still starting: keep the stock fields, but override the state so the server keeps waiting.
        return {**status, "state": "Pending", "reason": "NEMO_SERVICES_STARTING", "message": entry.message}

    def find_pod(self, sandbox_id: str, namespace: str | None = None) -> Any:
        """The sandbox's pod as a kubernetes ``V1Pod``, found by the stock sandbox-id label; None if it has none yet."""
        pods = self.raw_k8s_client.list_pods(
            namespace=namespace or self.namespace, label_selector=f"{SANDBOX_ID_LABEL}={sandbox_id}"
        )
        return pods[0] if pods else None

    def _tracked(self, sandbox_id: str) -> _Creating | None:
        """The sandbox's status-check state, or None if it isn't tracked. Drops the state once past its deadline."""
        with self._creating_lock:
            entry = self._creating.get(sandbox_id)
            if entry is not None and time.monotonic() > entry.deadline:
                del self._creating[sandbox_id]
                return None
            return entry

    def _forget(self, sandbox_id: str) -> None:
        """Stop tracking the sandbox; from now on its status is the stock status."""
        with self._creating_lock:
            self._creating.pop(sandbox_id, None)

    def _check(self, sandbox_id: str, namespace: str | None, entry: _Creating) -> list[str]:
        """Read the sandbox's pod and return what it still waits for.

        Records a service failure, with the failed container's last log lines, on
        ``entry.failure``. A pod that can't be read, or doesn't exist yet, counts as still
        starting, so a brief API error doesn't fail the create.
        """
        try:
            pod = self.find_pod(sandbox_id, namespace)
        except Exception:
            logger.warning("nemo services: cannot read the pod of sandbox %s", sandbox_id, exc_info=True)
            return ["pod status"]

        # The BatchSandbox controller creates the pod shortly after the BatchSandbox itself.
        if pod is None:
            return ["pod"]

        failure = service_failure(pod, entry.roles)
        if failure is None:
            return services_waiting(pod, entry.roles)

        # Put the failed container's last log lines in the message: they usually say why it failed.
        message = f"service {failure.container!r} failed ({failure.detail})."
        tail = self._log_tail(pod, failure.container)
        if tail:
            message += f" Last log lines:\n{tail}"

        logger.warning("nemo services: sandbox %s: %s", sandbox_id, message)
        entry.failure = message
        return []

    def _log_tail(self, pod: Any, container: str) -> str:
        """The container's last ``LOG_TAIL_LINES`` log lines, or a placeholder if they can't be read.

        Reading logs is best effort: the failure is reported either way.
        """
        try:
            return self.raw_k8s_client.get_core_v1_api().read_namespaced_pod_log(
                pod.metadata.name, pod.metadata.namespace, container=container, tail_lines=LOG_TAIL_LINES
            )
        except Exception as exc:
            return f"<logs unavailable: {exc.__class__.__name__}>"


# ---- HTTP routes -------------------------------------------------------------------------


class ExecRequest(BaseModel):
    """Body of the exec route: the argv to run, without a shell, and how many seconds to wait for it."""

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    timeout: float = Field(default=300, gt=0, le=3600)


class ExecResponse(BaseModel):
    """Result of the exec route. ``exit_code`` is -1 when Kubernetes reported none."""

    exit_code: int
    stdout: str
    stderr: str


def _not_found(message: str) -> HTTPException:
    """A 404 in the server's error shape."""
    return HTTPException(status_code=404, detail={"code": "NEMO::NOT_FOUND", "message": message})


def _require_provider() -> NemoServicesProvider:
    """The active provider, or a 503 when the server runs another runtime and never built one."""
    if _active_provider is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "NEMO::EXTENSION_INACTIVE", "message": "the server is not using the batchsandbox runtime"},
        )
    return _active_provider


def _read_exec_output(ws: Any, timeout: float) -> tuple[str, str]:
    """Read an exec stream's stdout and stderr until the command exits and the stream closes.

    Raises a 504 if the command runs past ``timeout`` seconds, and a 413 if its combined output
    passes ``EXEC_OUTPUT_LIMIT_BYTES``. The caller closes the stream either way, which ends the
    command's exec session.
    """
    out: list[str] = []
    err: list[str] = []
    size = 0
    deadline = time.monotonic() + timeout

    while ws.is_open():
        if time.monotonic() > deadline:
            raise HTTPException(
                status_code=504,
                detail={"code": "NEMO::EXEC_TIMEOUT", "message": f"command timed out after {timeout}s"},
            )

        # Wait up to a second for new frames, then drain whatever arrived on each channel.
        ws.update(timeout=1)
        for peek, read, sink in ((ws.peek_stdout, ws.read_stdout, out), (ws.peek_stderr, ws.read_stderr, err)):
            if peek():
                chunk = read()
                size += len(chunk)
                sink.append(chunk)

        if size > EXEC_OUTPUT_LIMIT_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "NEMO::EXEC_OUTPUT_TOO_LARGE",
                    "message": f"command output exceeds {EXEC_OUTPUT_LIMIT_BYTES} bytes",
                },
            )

    return "".join(out), "".join(err)


def run_exec(provider: NemoServicesProvider, sandbox_id: str, container: str, req: ExecRequest) -> ExecResponse:
    """Run ``req.command`` in one of the sandbox's service containers through the Kubernetes exec API.

    Returns 404 for a sandbox without a pod, and for any container that isn't one of its
    services. Blocks until the command exits, so the route runs it on a worker thread.
    """
    pod = provider.find_pod(sandbox_id)
    if pod is None:
        raise _not_found(f"sandbox {sandbox_id} has no pod")

    # Only containers this extension added: never main, egress or execd. The roles annotation
    # lists exactly those, so a pod created without services has none.
    if container not in pod_roles(pod):
        raise _not_found(f"{container!r} is not a service of sandbox {sandbox_id}")

    # stream() swaps the request function of the ApiClient it runs on, so never share one
    # between concurrent execs.
    api_client = k8s.ApiClient()
    try:
        ws = stream(
            k8s.CoreV1Api(api_client).connect_get_namespaced_pod_exec,
            pod.metadata.name,
            pod.metadata.namespace,
            container=container,
            command=req.command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            stdout, stderr = _read_exec_output(ws, req.timeout)
            code = ws.returncode
        finally:
            ws.close()
    finally:
        api_client.close()

    return ExecResponse(exit_code=code if code is not None else -1, stdout=stdout, stderr=stderr)


router = APIRouter(prefix="/v1")


@router.post("/sandboxes/{sandbox_id}/containers/{container}/exec", response_model=ExecResponse)
async def exec_in_service(sandbox_id: str, container: str, req: ExecRequest) -> ExecResponse:
    """HTTP route for ``run_exec``. The blocking Kubernetes stream runs on a worker thread, off the event loop."""
    provider = _require_provider()
    return await asyncio.to_thread(run_exec, provider, sandbox_id, container, req)


@router.get("/nemo-ext/health")
async def health() -> dict[str, Any]:
    """Lets clients check that this extension is loaded and active before they create a sandbox with services."""
    _require_provider()
    return {"status": "ok", "extension": "nemo-services"}


# ---- install -----------------------------------------------------------------------------


def register_provider() -> None:
    """Make the server build ``NemoServicesProvider`` for the batchsandbox runtime.

    The server builds its provider once, while ``opensandbox_server.main`` loads, so this must
    run before that; ``sitecustomize.py`` takes care of the timing.
    """
    provider_factory.register_provider(provider_factory.PROVIDER_TYPE_BATCHSANDBOX, NemoServicesProvider)
    logger.info("nemo services: registered NemoServicesProvider as %r", provider_factory.PROVIDER_TYPE_BATCHSANDBOX)


def attach_routes(app: FastAPI) -> None:
    """Add the exec and health routes to the server's app. Its API-key middleware already covers every route."""
    app.include_router(router)
    logger.info("nemo services: added %s", ", ".join(sorted(getattr(r, "path", "?") for r in router.routes)))
