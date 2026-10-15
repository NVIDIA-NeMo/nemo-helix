# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface of the trusted episode provisioning broker.

The job sandbox reaches the broker over HTTP and never as a Ray client: joining the training Ray
cluster is exactly what the surrounding design forbids, since a GCS address is cluster-wide
scheduling authority.

This module imports no Ray and no training code, so it can be exercised without a cluster. Ray
placement and process lifecycle live in :mod:`sandboxed_gym.broker_actor`.
"""

import base64
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sandboxed_gym import audit
from sandboxed_gym.backends.base import (
    DirectoryTransferError,
    EpisodeSandboxBackend,
    UnsupportedEpisodeOperationError,
)
from sandboxed_gym.config import EpisodeBrokerConfig
from sandboxed_gym.episodes import EpisodeRegistry
from sandboxed_gym.errors import BrokerRequestError
from sandboxed_gym.sanitize import (
    sanitize_create_request,
    sanitize_exec_request,
)
from sandboxed_gym.wire import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    EPISODE_DIRS_PATH,
    EPISODE_EXEC_PATH,
    EPISODE_FILES_PATH,
    EPISODE_PATH,
    EPISODES_PATH,
    HEALTH_PATH,
    BrokerErrorCode,
    BrokerErrorResponse,
    BrokerHealthResponse,
    EpisodeCloseResponse,
    EpisodeCreateRequest,
    EpisodeCreateResponse,
    EpisodeDirDownloadResponse,
    EpisodeDirUploadRequest,
    EpisodeExecRequest,
    EpisodeExecResponse,
    EpisodeFileDownloadResponse,
    EpisodeFileUploadRequest,
    EpisodeStatusResponse,
    validate_absolute_path,
)

LOGGER = logging.getLogger(__name__)


class _PayloadTooLargeError(Exception):
    """Raised inside the ASGI receive chain when a request body exceeds the cap."""


def _error_body(message: str, code: BrokerErrorCode) -> dict[str, str]:
    return BrokerErrorResponse(error=message, code=code).model_dump(mode="json")


def _declared_content_length(scope: dict) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes``.

    The broker buffers request bodies in the *trusted* training leader process, so an unbounded
    upload from the job sandbox is a denial-of-service path into the trusted plane rather than into
    the sandbox. The declared ``Content-Length`` is checked first, and the received body is counted
    as it streams so a chunked request cannot slip past.
    """

    def __init__(self, app, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def _send_too_large(self, send) -> None:
        body = json.dumps(
            _error_body(
                f"request body exceeds the {self._max_bytes} byte limit",
                BrokerErrorCode.PAYLOAD_TOO_LARGE,
            )
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        """Enforce the body cap for HTTP requests, passing other scopes through."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        declared = _declared_content_length(scope)
        if declared is not None and declared > self._max_bytes:
            await self._send_too_large(send)
            return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    raise _PayloadTooLargeError
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, limited_receive, tracking_send)
        except _PayloadTooLargeError:
            if response_started:
                raise
            await self._send_too_large(send)


@contextlib.asynccontextmanager
async def _backend_errors(operation: str) -> AsyncGenerator[None, None]:
    """Translate backend failures into wire errors without disclosing backend internals.

    The full exception is logged on the trusted side; the caller gets a generic message. An
    unsupported operation is the exception -- that message is ours and telling the environment
    exactly what is unsupported is the point, so it fails fast instead of grading under conditions
    it did not ask for.
    """
    try:
        yield
    except BrokerRequestError:
        raise
    except UnsupportedEpisodeOperationError as e:
        raise BrokerRequestError(BrokerErrorCode.UNSUPPORTED_OPERATION, str(e), status_code=501) from e
    except DirectoryTransferError as e:
        # Also ours to disclose: the cause is the episode image (no `tar`, unwritable /tmp), which
        # the caller can fix and which says nothing about the backend.
        LOGGER.warning("Directory transfer failed during %s: %s", operation, e)
        raise BrokerRequestError(BrokerErrorCode.BACKEND_ERROR, str(e), status_code=502) from e
    except Exception as e:
        LOGGER.exception("Episode backend failed during %s", operation)
        raise BrokerRequestError(
            BrokerErrorCode.BACKEND_ERROR,
            f"episode backend error during {operation}",
            status_code=502,
        ) from e


def build_broker_app(
    *,
    backend: EpisodeSandboxBackend,
    config: EpisodeBrokerConfig,
    token: str,
) -> FastAPI:
    """Build the broker's FastAPI application.

    Args:
        backend: Episode backend holding whatever credential it needs. Trusted-side only.
        config: Broker policy for this job.
        token: Job-scoped bearer token the job sandbox must present. It is readable by user code
            by design, so it identifies the job rather than granting capability -- every operation
            reachable with it has to be escalation-free on its own.

    Returns:
        A configured :class:`fastapi.FastAPI` application.
    """
    episodes = EpisodeRegistry(max_concurrent=config.max_concurrent_episodes)

    app = FastAPI(
        title="NeMo-RL episode provisioning broker",
        version=BROKER_PROTOCOL_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.max_request_bytes)
    app.state.backend = backend
    app.state.config = config
    app.state.episodes = episodes
    # Set before teardown drains, so nothing new can be provisioned behind the drain. See
    # begin_shutdown.
    app.state.closing = False

    def _authenticate(request: Request) -> None:
        provided = request.headers.get(BROKER_AUTH_HEADER)
        if provided is None or not hmac.compare_digest(provided.encode("utf-8"), token.encode("utf-8")):
            raise BrokerRequestError(
                BrokerErrorCode.UNAUTHORIZED,
                "missing or invalid broker token",
                status_code=401,
            )

    auth = Depends(_authenticate)

    @app.exception_handler(BrokerRequestError)
    async def _on_broker_error(request: Request, exc: BrokerRequestError) -> JSONResponse:
        # Every refusal funnels through here, so this is the one place that has to record them.
        # A backend outage or a shutdown is not a policy denial; keeping them apart is what lets
        # someone filter the log for "what did this environment try that we refused".
        failed = exc.code in (
            BrokerErrorCode.BACKEND_ERROR,
            BrokerErrorCode.SHUTTING_DOWN,
        )
        audit.record(
            "request.rejected" if not failed else "request.failed",
            job_id=config.job_id,
            outcome="failed" if failed else "rejected",
            method=request.method,
            path=request.url.path,
            code=exc.code.value,
            status=exc.status_code,
            reason=audit.truncate(exc.message),
        )
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.message, exc.code))

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Echoing the caller's own malformed fields back is safe and is what makes a rejected
        # environment debuggable.
        summary = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
            for error in exc.errors()[:10]
        )
        audit.record(
            "request.rejected",
            job_id=config.job_id,
            outcome="rejected",
            method=request.method,
            path=request.url.path,
            code=BrokerErrorCode.INVALID_REQUEST.value,
            status=422,
            reason=audit.truncate(summary),
        )
        return JSONResponse(
            status_code=422,
            content=_error_body(f"invalid request: {summary}", BrokerErrorCode.INVALID_REQUEST),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(str(exc.detail), BrokerErrorCode.INVALID_REQUEST),
        )

    @app.get(HEALTH_PATH, response_model=BrokerHealthResponse, dependencies=[auth])
    async def health() -> BrokerHealthResponse:
        """Report broker readiness and the protocol version the client must match."""
        return BrokerHealthResponse(job_id=config.job_id)

    @app.post(
        EPISODES_PATH,
        response_model=EpisodeCreateResponse,
        status_code=201,
        dependencies=[auth],
    )
    async def create_episode(request: EpisodeCreateRequest) -> EpisodeCreateResponse:
        """Create one episode sandbox from a sanitized spec."""
        if app.state.closing:
            # Refusing here is what makes teardown correct rather than merely conventional: an
            # episode created behind the drain would never be reaped.
            raise BrokerRequestError(
                BrokerErrorCode.SHUTTING_DOWN,
                "broker is shutting down and will not create new episodes",
                status_code=503,
            )
        spec = sanitize_create_request(request, config)
        episode_id = await episodes.reserve()

        try:
            async with _backend_errors("create"):
                backend_id = await backend.create(spec)
        except BaseException:
            await episodes.release(episode_id)
            raise

        try:
            async with _backend_errors("file staging"):
                for path, content in spec.files.items():
                    await backend.upload_file(backend_id, path, content)
        except BaseException:
            await episodes.release(episode_id)
            with contextlib.suppress(Exception):
                await backend.close(backend_id)
            raise

        await episodes.bind(episode_id, backend_id)
        audit.record(
            "episode.create",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            image=spec.image,
            ttl_s=spec.ttl_s,
            staged_files=len(spec.files),
            egress_default_action=backend.egress.default_action,
        )
        return EpisodeCreateResponse(episode_id=episode_id)

    @app.get(EPISODE_PATH, response_model=EpisodeStatusResponse, dependencies=[auth])
    async def episode_status(episode_id: str) -> EpisodeStatusResponse:
        """Report one episode's lifecycle status."""
        backend_id = await episodes.resolve(episode_id)
        async with _backend_errors("status"):
            return EpisodeStatusResponse(status=await backend.status(backend_id))

    @app.post(EPISODE_EXEC_PATH, response_model=EpisodeExecResponse, dependencies=[auth])
    async def episode_exec(episode_id: str, request: EpisodeExecRequest) -> EpisodeExecResponse:
        """Run one command inside an episode."""
        backend_id = await episodes.resolve(episode_id)
        call = sanitize_exec_request(request, config)
        async with _backend_errors("exec"):
            result = await backend.exec(
                backend_id,
                call.command,
                cwd=call.cwd,
                env=call.env or None,
                timeout_s=call.timeout_s,
                user=call.user,
            )
        # Command text is recorded (truncated) because "what did this environment run as root"
        # is the question an audit of a grading run has to answer. Environment values are not.
        audit.record(
            "episode.exec",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            user=call.user,
            timeout_s=call.timeout_s,
            env_keys=sorted(call.env),
            command=audit.truncate(call.command),
            return_code=result.return_code,
        )
        return EpisodeExecResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            error_type=result.error_type,
        )

    @app.put(EPISODE_FILES_PATH, status_code=204, dependencies=[auth])
    async def upload_episode_file(episode_id: str, request: EpisodeFileUploadRequest) -> Response:
        """Write one file into an episode."""
        backend_id = await episodes.resolve(episode_id)
        content = base64.b64decode(request.content_b64)
        async with _backend_errors("upload"):
            await backend.upload_file(backend_id, request.path, content)
        audit.record(
            "episode.file.upload",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            path=request.path,
            bytes=len(content),
        )
        return Response(status_code=204)

    @app.get(
        EPISODE_FILES_PATH,
        response_model=EpisodeFileDownloadResponse,
        dependencies=[auth],
    )
    async def download_episode_file(episode_id: str, path: str) -> EpisodeFileDownloadResponse:
        """Read one file out of an episode."""
        backend_id = await episodes.resolve(episode_id)
        try:
            validate_absolute_path(path)
        except ValueError as e:
            raise BrokerRequestError(BrokerErrorCode.INVALID_REQUEST, str(e), status_code=400) from e

        async with _backend_errors("download"):
            content = await backend.download_file(backend_id, path)

        # Backends hand back whole files today, so this bounds what the broker will *serialize*
        # rather than what it reads. Bounding the read itself needs backend-side support.
        if len(content) > config.max_request_bytes:
            raise BrokerRequestError(
                BrokerErrorCode.PAYLOAD_TOO_LARGE,
                f"file exceeds the {config.max_request_bytes} byte transfer limit: {path}",
                status_code=413,
            )
        audit.record(
            "episode.file.download",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            path=path,
            bytes=len(content),
        )
        return EpisodeFileDownloadResponse(content_b64=base64.b64encode(content).decode())

    @app.put(EPISODE_DIRS_PATH, status_code=204, dependencies=[auth])
    async def upload_episode_dir(episode_id: str, request: EpisodeDirUploadRequest) -> Response:
        """Unpack a directory tree into an episode."""
        backend_id = await episodes.resolve(episode_id)
        archive = base64.b64decode(request.archive_b64)
        async with _backend_errors("directory upload"):
            await backend.upload_dir(backend_id, request.path, archive)
        audit.record(
            "episode.dir.upload",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            path=request.path,
            bytes=len(archive),
        )
        return Response(status_code=204)

    @app.get(
        EPISODE_DIRS_PATH,
        response_model=EpisodeDirDownloadResponse,
        dependencies=[auth],
    )
    async def download_episode_dir(episode_id: str, path: str) -> EpisodeDirDownloadResponse:
        """Read a directory tree out of an episode as a gzipped tar."""
        backend_id = await episodes.resolve(episode_id)
        try:
            validate_absolute_path(path)
        except ValueError as e:
            raise BrokerRequestError(BrokerErrorCode.INVALID_REQUEST, str(e), status_code=400) from e

        async with _backend_errors("directory download"):
            archive = await backend.download_dir(backend_id, path)

        # Same bound, and the same caveat, as the single-file download: this limits what the broker
        # will serialize, not what the backend read. A directory is the likelier one to hit it.
        if len(archive) > config.max_request_bytes:
            raise BrokerRequestError(
                BrokerErrorCode.PAYLOAD_TOO_LARGE,
                f"directory archive exceeds the {config.max_request_bytes} byte transfer limit: {path}",
                status_code=413,
            )
        audit.record(
            "episode.dir.download",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
            path=path,
            bytes=len(archive),
        )
        return EpisodeDirDownloadResponse(archive_b64=base64.b64encode(archive).decode())

    @app.delete(EPISODE_PATH, response_model=EpisodeCloseResponse, dependencies=[auth])
    async def close_episode(episode_id: str) -> EpisodeCloseResponse:
        """Terminate one episode and free its concurrency slot."""
        # The slot is freed before the backend call so a backend that fails to tear an episode down
        # cannot permanently consume the job's episode budget. TTL and the reaper cover the leak.
        backend_id = await episodes.release(episode_id)
        if backend_id is None:
            raise BrokerRequestError(
                BrokerErrorCode.EPISODE_NOT_FOUND,
                f"unknown episode: {episode_id}",
                status_code=404,
            )
        async with _backend_errors("close"):
            await backend.close(backend_id)
        audit.record(
            "episode.close",
            job_id=config.job_id,
            outcome="allowed",
            episode_id=episode_id,
        )
        return EpisodeCloseResponse()

    return app


def begin_shutdown(app: FastAPI) -> None:
    """Stop accepting new episodes, without disturbing the ones already running.

    Call this before :func:`close_all_episodes`. Draining while the listener still accepts creates
    leaves a window where the job sandbox can provision an episode the drain has already walked
    past, which would then never be reaped. Operating and closing existing episodes keeps working,
    because teardown depends on it.
    """
    app.state.closing = True


async def close_all_episodes(app: FastAPI) -> None:
    """Terminate every episode the broker still owns, then release backend resources.

    Called during teardown. Failures are logged rather than raised so one stuck episode cannot
    block the rest of shutdown; backend TTL is the backstop.
    """
    backend: EpisodeSandboxBackend = app.state.backend
    episodes: EpisodeRegistry = app.state.episodes

    for backend_id in await episodes.drain():
        try:
            await backend.close(backend_id)
        except Exception:
            LOGGER.exception("Failed to close episode %s during broker shutdown", backend_id)

    try:
        await backend.aclose()
    except Exception:
        LOGGER.exception("Failed to release episode backend resources during broker shutdown")
