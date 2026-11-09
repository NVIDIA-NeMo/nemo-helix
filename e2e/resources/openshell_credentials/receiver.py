# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe credentials injected into synthetic sandbox requests without exposing them."""

import argparse
import hashlib
import hmac
import threading
import time
from collections import deque
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr, ValidationError
from starlette.middleware.base import RequestResponseEndpoint


class ProbeReport(BaseModel):
    probe_id: str = Field(min_length=1, max_length=128)
    pid: int = Field(strict=True, gt=0)
    boot_id: str = Field(min_length=32, max_length=32)
    attempt: int = Field(strict=True, gt=0)
    placeholder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forbidden_status: Annotated[int, Field(strict=True, ge=100, le=599)] | None
    forbidden_error: Annotated[str, Field(max_length=80)] | None
    public_status: Annotated[int, Field(strict=True, ge=100, le=599)] | None


class Observation(ProbeReport):
    sequence: int
    received_at: float
    accepted: bool
    version: str | None


class Heartbeat(BaseModel):
    probe_id: str
    pid: int
    boot_id: str
    placeholder_sha256: str
    completed_attempt: int
    probe_status: int | None
    probe_error: str | None


class ExpectedTokens(BaseModel):
    tokens: dict[
        Annotated[str, Field(min_length=1, max_length=64)],
        Annotated[SecretStr, Field(min_length=1, max_length=4096)],
    ] = Field(max_length=16)


class ObservationPage(BaseModel):
    observations: list[Observation]
    latest_sequence: int
    heartbeats: list[Heartbeat]


class ReceiverState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.expected: dict[str, str] = {}
        self.observations: deque[Observation] = deque(maxlen=256)
        self.sequence = 0
        self.heartbeats: dict[str, Heartbeat] = {}

    def record(self, authorization: str, report: ProbeReport) -> Observation:
        digest = hashlib.sha256(authorization.encode()).hexdigest()
        with self.lock:
            version = next((name for name, value in self.expected.items() if hmac.compare_digest(value, digest)), None)
            self.sequence += 1
            observation = Observation(
                **report.model_dump(),
                sequence=self.sequence,
                received_at=time.time(),
                accepted=version is not None,
                version=version,
            )
            self.observations.append(observation)
            return observation


def create_app() -> FastAPI:
    app = FastAPI(openapi_url=None)
    state = ReceiverState()

    @app.middleware("http")
    async def limit_request_body(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST":
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= 16384:
                return JSONResponse(
                    status_code=400, content={"error": "POST bodies must be between 1 and 16384 bytes."}
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # Validation errors can include submitted tokens in their input and location.
        return JSONResponse(
            status_code=400,
            content={"errors": [{"type": item["type"], "message": item["msg"]} for item in error.errors()]},
        )

    @app.get("/health")
    def health(heartbeat: str | None = None) -> dict[str, bool]:
        if heartbeat is not None:
            try:
                report = Heartbeat.model_validate_json(heartbeat)
            except ValidationError as error:
                raise RequestValidationError(error.errors()) from None
            with state.lock:
                state.heartbeats[report.probe_id] = report
        return {"ready": True}

    @app.get("/observations")
    def observations(after: int = 0) -> ObservationPage:
        with state.lock:
            return ObservationPage(
                observations=[row for row in state.observations if row.sequence > after],
                latest_sequence=state.sequence,
                heartbeats=list(state.heartbeats.values()),
            )

    @app.post("/expect")
    def expect(body: ExpectedTokens) -> dict[str, list[str]]:
        expected = {
            version: hashlib.sha256(f"Bearer {token.get_secret_value()}".encode()).hexdigest()
            for version, token in body.tokens.items()
        }
        with state.lock:
            state.expected = expected
        return {"versions": list(expected)}

    @app.post("/probe")
    def probe(report: ProbeReport, response: Response, authorization: Annotated[str, Header()] = "") -> Observation:
        observation = state.record(authorization, report)
        response.status_code = 200 if observation.accepted else 401
        return observation

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    args = parser.parse_args()
    if bool(args.cert_file) != bool(args.key_file):
        parser.error("--cert-file and --key-file must be supplied together")
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        ssl_certfile=args.cert_file,
        ssl_keyfile=args.key_file,
        access_log=False,
    )


if __name__ == "__main__":
    main()
