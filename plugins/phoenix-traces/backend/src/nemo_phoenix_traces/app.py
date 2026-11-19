# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent FastAPI adapter. Never imported or launched by the Helix registry."""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client.tls import client_verify_from_env
from nemo_helix_plugin.community_plugins.registry import (
    BundleSpec,
    EndpointMapping,
    ExternalPlugin,
    NavigationItem,
    StudioSidecar,
    WorkspaceSecretCredential,
    validate_remote_url,
)
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import DirectSpansIngestRequest
from nemo_phoenix_traces.models import PhoenixSpan, Project, ProjectsPage, SpanBatch, SpansPage
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOENIX_", extra="ignore", hide_input_in_errors=True)
    workspace: str = Field(default="default", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    adapter_key: SecretStr = Field(min_length=24, max_length=4096)
    intake_url: str = "http://127.0.0.1:8080"
    intake_token: SecretStr | None = None
    public_url: str = "http://127.0.0.1:6007"
    secret_name: str = "phoenix-traces-adapter-key"
    bundle_dir: Path = Path(__file__).resolve().parents[3] / "dist"

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        for url in (self.intake_url, self.public_url):
            validate_remote_url(url)
            if urlsplit(url).path not in {"", "/"}:
                raise ValueError("Service URLs must be origins without a path")
        key = self.adapter_key.get_secret_value()
        if any(not 33 <= ord(char) <= 126 for char in key):
            raise ValueError("Adapter key must contain only visible ASCII characters")
        if self.intake_token and key == self.intake_token.get_secret_value():
            raise ValueError("Adapter and Intake credentials must be distinct")
        WorkspaceSecretCredential(secret_name=self.secret_name)
        return self


class Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1, le=10000)
    limit: int = Field(ge=1, le=1000)
    workspace: str
    project_sha256: str | None

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")


def _project_digest(project: str | None) -> str | None:
    return hashlib.sha256(project.encode()).hexdigest() if project is not None else None


def page_number(cursor: str | None, *, limit: int, workspace: str, project: str | None) -> int:
    if cursor is None:
        return 1
    try:
        decoded = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        state = Cursor.model_validate_json(decoded)
        if (state.limit, state.workspace, state.project_sha256) != (limit, workspace, _project_digest(project)):
            raise ValueError("Cursor belongs to another query")
        return state.page
    except (ValueError, ValidationError) as exc:
        raise HTTPException(400, "Invalid cursor for this query") from exc


def next_cursor(*, page: int, total_pages: int, limit: int, workspace: str, project: str | None) -> str | None:
    if page >= total_pages:
        return None
    if page >= 10000:
        raise HTTPException(422, "Query exceeds the adapter pagination limit")
    return Cursor(page=page + 1, limit=limit, workspace=workspace, project_sha256=_project_digest(project)).encode()


def plugin_manifest(settings: Settings) -> dict:
    base = settings.public_url.rstrip("/")
    endpoints = tuple(
        EndpointMapping(
            method=method,
            path=f"/v2/workspaces/{{workspace}}/{path}",
            access="read" if method == "GET" else "write",
            upstream_url=f"{base}/v1/workspaces/{{workspace}}/{path}",
        )
        for method, path in (("GET", "projects"), ("GET", "spans"), ("POST", "spans"))
    )
    bundle = None
    navigation: tuple[NavigationItem, ...] = ()
    script = settings.bundle_dir / "index.js"
    sidecar = settings.bundle_dir / "plugin.manifest.json"
    if script.is_file() and sidecar.is_file():
        bundle = BundleSpec(
            url=f"{base}/index.js",
            sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            sidecar=StudioSidecar.model_validate_json(sidecar.read_bytes()),
        )
        navigation = (NavigationItem(label="Phoenix", path="/workspaces/{workspace}/plugin/phoenix"),)
    plugin = ExternalPlugin(
        name="phoenix",
        title="Phoenix traces (Intake)",
        endpoints=endpoints,
        credential=WorkspaceSecretCredential(secret_name=settings.secret_name),
        bundle=bundle,
        navigation=navigation,
    )
    return {"schema_version": 1, "plugin": plugin.model_dump(mode="json", by_alias=True, exclude_none=True)}


def create_app(settings: Settings | None = None, *, intake_client: AsyncIntakeClient | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if intake_client is not None:
            app.state.intake = intake_client
            yield
            return
        headers = (
            {"Authorization": f"Bearer {settings.intake_token.get_secret_value()}"} if settings.intake_token else {}
        )
        # Do not inherit caller headers, environment proxy settings, or platform
        # identity. This operator-provisioned credential is fixed to this origin.
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=False, trust_env=False, verify=client_verify_from_env()
        ) as transport:
            async with AsyncNemoClient(
                base_url=settings.intake_url.rstrip("/"), default_headers=headers, http_client=transport
            ) as client:
                app.state.intake = AsyncIntakeClient.from_client(client)
                yield

    app = FastAPI(title="Phoenix / Intake adapter", lifespan=lifespan, docs_url=None, redoc_url=None)

    async def require_access(request: Request, workspace: str) -> None:
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {settings.adapter_key.get_secret_value()}"
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "Adapter credential required", headers={"WWW-Authenticate": "Bearer"})
        if workspace != settings.workspace:
            raise HTTPException(403, "This adapter is bound to a different workspace")

    @app.exception_handler(NemoClientError)
    async def intake_error(_request: Request, error: NemoClientError) -> JSONResponse:
        # Never return an upstream body: it can contain trace data or credentials.
        status = 503 if isinstance(error, NemoTransportError) else 502
        if isinstance(error, NemoHTTPError) and error.status_code in {400, 403, 404, 409, 413, 422, 429, 503}:
            status = error.status_code
        return JSONResponse({"detail": "Intake request failed; check adapter configuration and Intake health"}, status)

    @app.get("/plugin.yaml")
    async def manifest() -> Response:
        return Response(yaml.safe_dump(plugin_manifest(settings), sort_keys=False), media_type="application/yaml")

    @app.get("/index.js")
    async def script() -> Response:
        path = settings.bundle_dir / "index.js"
        if not path.is_file():
            raise HTTPException(404, "Build the plugin web bundle first")
        return Response(path.read_bytes(), media_type="text/javascript", headers={"Cache-Control": "no-store"})

    access = [Depends(require_access)]

    @app.get("/v1/workspaces/{workspace}/projects", dependencies=access)
    async def projects(
        request: Request,
        workspace: str,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        cursor: Annotated[str | None, Query(max_length=4096)] = None,
    ) -> ProjectsPage:
        page = page_number(cursor, limit=limit, workspace=workspace, project=None)
        client: AsyncIntakeClient = request.app.state.intake
        result = await client.list_span_groups(
            workspace=settings.workspace,
            query_params={"by": "project", "page": page, "page_size": limit, "sort": "started_at"},
        )
        data = result.page()
        return ProjectsPage(
            data=[Project(name=item.group.get("project", "")) for item in data.items],
            next_cursor=next_cursor(
                page=page, total_pages=data.metadata["total_pages"], limit=limit, workspace=workspace, project=None
            ),
        )

    @app.get("/v1/workspaces/{workspace}/spans", dependencies=access)
    async def spans(
        request: Request,
        workspace: str,
        project: Annotated[str, Query(max_length=4096)],
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        cursor: Annotated[str | None, Query(max_length=4096)] = None,
    ) -> SpansPage:
        page = page_number(cursor, limit=limit, workspace=workspace, project=project)
        client: AsyncIntakeClient = request.app.state.intake
        result = await client.list_spans(
            workspace=settings.workspace,
            query_params={
                "filter": {"project": project},
                "page": page,
                "page_size": limit,
                "sort": "started_at",
            },
        )
        data = result.page()
        try:
            projected = [PhoenixSpan.from_intake(span) for span in data.items]
        except (ValueError, ValidationError) as exc:
            raise HTTPException(502, "Intake returned invalid span data") from exc
        return SpansPage(
            data=projected,
            next_cursor=next_cursor(
                page=page, total_pages=data.metadata["total_pages"], limit=limit, workspace=workspace, project=project
            ),
        )

    @app.post("/v1/workspaces/{workspace}/spans", dependencies=access, status_code=201)
    async def ingest(
        request: Request,
        workspace: str,
        project: Annotated[str, Query(max_length=4096)],
        body: SpanBatch,
    ) -> Response:
        client: AsyncIntakeClient = request.app.state.intake
        await client.create_spans(
            workspace=settings.workspace,
            body=DirectSpansIngestRequest(
                source="phoenix-plugin", spans=[span.to_intake(project) for span in body.spans]
            ),
        )
        return Response(status_code=201)

    return app
