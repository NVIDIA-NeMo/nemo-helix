# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local FastAPI proxy: /health and /rollouts/run → Gym host."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from sandboxed_gym.orchestrator import SandboxedGymSession
from sandboxed_gym.wire import PROXY_AUTH_HEADER

LOGGER = logging.getLogger(__name__)

#: Re-exported under its original name; the constant itself lives in the dependency-free wire module.
AUTH_HEADER = PROXY_AUTH_HEADER

#: Upper bound on one forwarded read, not a batch size: ``read1`` returns whatever has already
#: arrived, so a lone heartbeat byte is passed straight through rather than waiting for a full one.
_STREAM_CHUNK_BYTES = 64 * 1024


async def _forward(response: Any, max_response_bytes: int) -> AsyncGenerator[bytes, None]:
    """Yield the upstream body as it arrives, so a slow rollout stays visibly alive downstream.

    The Gym host answers ``200`` before its batch finishes and pads the body with whitespace while
    it works, so that no hop mistakes a long rollout for a dead one. Reading the response to
    completion here would absorb every one of those bytes and re-emit them at the end, leaving this
    proxy's own caller with exactly the silence the padding exists to prevent.

    ``read1`` rather than ``read``: ``read`` blocks until the requested count or EOF, which would
    reintroduce the buffering one chunk at a time.

    Async rather than a plain generator, even though every read is blocking. Starlette runs a
    *sync* iterator through ``iterate_in_threadpool``, and a client disconnect cannot interrupt a
    worker thread parked in ``read1`` -- the ``finally`` below would not run until the host next
    produced a byte, holding the upstream connection open for the rest of the batch. Awaiting each
    read instead gives cancellation somewhere to land, and closing the response is what frees the
    thread still sitting in that read.
    """
    forwarded = 0
    try:
        while True:
            chunk = await asyncio.to_thread(response.read1, _STREAM_CHUNK_BYTES)
            if not chunk:
                return
            forwarded += len(chunk)
            if forwarded > max_response_bytes:
                # Raised, not returned. The status line is long gone, so there is no 502 left to
                # send -- but ending the stream cleanly here would hand the caller a 200 carrying a
                # prefix of the body, and a prefix of `{"results": [...]}` can parse as a smaller
                # but perfectly valid result set. A silently short rollout batch is worse than a
                # failed one. Raising aborts the response mid-body instead, which every HTTP client
                # surfaces as a broken read.
                raise RuntimeError(
                    f"upstream rollout response exceeded max_response_bytes "
                    f"({forwarded} > {max_response_bytes}); aborting rather than forwarding a "
                    f"truncated body that could parse as a complete one"
                )
            yield chunk
    finally:
        # Runs on cancellation too, which is the point: a disconnect closes the upstream instead of
        # leaving it open until the rollout deadline.
        response.close()


def build_proxy_app(session: SandboxedGymSession) -> FastAPI:
    """Proxy Gym host endpoints; optionally require ``rollout_auth_token``."""
    app = FastAPI(title="sandboxed-gym-orchestrator", version="0.1.0")
    expected = session.cfg.rollout_auth_token

    def _check_auth(token: str | None) -> None:
        if expected is None:
            return
        if not token or token != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health(
        x_sandboxed_gym_token: str | None = Header(default=None, alias=AUTH_HEADER),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        _check_auth(x_sandboxed_gym_token or bearer)
        request = urllib.request.Request(
            session.host.health_url,
            method="GET",
            headers=dict(session.host.headers),
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise HTTPException(status_code=exc.code, detail=exc.read().decode()) from exc
        except Exception as exc:
            # The caller is sandboxed job code, so the exception text stays server-side: it can name
            # the Gym host's internal address or carry request detail from a failed connection.
            LOGGER.exception("upstream health check failed")
            raise HTTPException(status_code=502, detail="upstream health check failed") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {"status": "unknown", "raw": body.decode("utf-8", errors="replace")}

    @app.post("/rollouts/run")
    async def rollouts_run(
        request: Request,
        x_sandboxed_gym_token: str | None = Header(default=None, alias=AUTH_HEADER),
        authorization: str | None = Header(default=None),
    ) -> Response:
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        _check_auth(x_sandboxed_gym_token or bearer)

        payload = await request.body()
        max_req = session.cfg.sandbox.max_request_bytes
        if len(payload) > max_req:
            raise HTTPException(status_code=413, detail="request too large")

        upstream = urllib.request.Request(
            session.host.rollout_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": request.headers.get("content-type", "application/json"),
                **session.host.headers,
            },
        )
        # Opened off the event loop, and awaited before streaming so the upstream status and
        # headers still decide this response's own -- once bytes start flowing neither can change.
        try:
            response = await asyncio.to_thread(
                urllib.request.urlopen, upstream, timeout=session.cfg.sandbox.rollout_timeout_s
            )
        except urllib.error.HTTPError as exc:
            return Response(
                content=exc.read(),
                status_code=exc.code,
                media_type="application/json",
            )
        except Exception:
            # Logged, not returned -- see the health handler above.
            LOGGER.exception("upstream rollout failed")
            return JSONResponse(status_code=502, content={"error": "upstream rollout failed"})

        return StreamingResponse(
            _forward(response, session.cfg.sandbox.max_response_bytes),
            status_code=response.status,
            media_type=response.headers.get("Content-Type", "application/json"),
        )

    @app.get("/session")
    def session_info(
        x_sandboxed_gym_token: str | None = Header(default=None, alias=AUTH_HEADER),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        _check_auth(x_sandboxed_gym_token or bearer)
        desc = session.descriptor(mode="orchestrator", orchestrator_url=session.orchestrator_url)
        data = desc.model_dump(mode="json")
        if expected is None:
            # Unauthenticated route: the caller is sandboxed job code, which must not read the
            # broker credential it is being brokered through.
            data.pop("broker_token", None)
            data.pop("rollout_auth_token", None)
        return data

    return app
