# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test fixtures for v2 entities API.

Provides async HTTP client fixtures for testing the v2 API endpoints.
"""

import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.dependencies import (
    dep_entity_repository_with_session,
    dep_workspace_repository_with_session,
)
from nhx.core.entities.api.server import app
from nhx.core.entities.app.repository import (
    SQLAlchemyEntityRepository,
    SQLAlchemyWorkspaceRepository,
)
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity, DBWorkspace  # noqa: F401
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.base import BaseHTTPMiddleware


def _entities_alembic_ini_path() -> Path:
    """Path to entities service alembic.ini (integration/ -> service root)."""
    return Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def _run_migrations_on_url(url: str, alembic_ini_path: Path | None = None) -> None:
    """Run Alembic upgrade head against the given database URL (test use only)."""
    ini = alembic_ini_path or _entities_alembic_ini_path()
    if not ini.is_file():
        raise RuntimeError(f"alembic.ini not found at {ini}; cannot run migrations")
    sync_url = url.replace("sqlite+aiosqlite", "sqlite").replace("postgresql+asyncpg", "postgresql")
    cfg = Config(str(ini))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    command.upgrade(cfg, "head")


# Test principal for authenticated requests
TEST_PRINCIPAL = "test-user@example.com"

# Service principal id for tests that need unscoped (all-workspace) access
TEST_SERVICE_PRINCIPAL = "service:entity-store"


def _enable_sqlite_fks(dbapi_conn, connection_record):
    """Enable FK constraints for SQLite connections."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


@pytest_asyncio.fixture
async def session_maker():
    """Create an async session maker with SQLite DB and Alembic migrations applied."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    sync_url = f"sqlite:///{db_path}"
    _run_migrations_on_url(sync_url, _entities_alembic_ini_path())
    async_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(async_url, echo=False)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_fks)
    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    try:
        yield session_maker
    finally:
        await engine.dispose()
        Path(db_path).unlink(missing_ok=True)


@pytest.fixture(params=["sqlalchemy"])
def repos(request, session_maker):
    """Parametrized fixture that provides repository implementations.

    The params list can be extended to test multiple repository backends.
    """
    if request.param == "sqlalchemy":
        return {
            "workspace": SQLAlchemyWorkspaceRepository(session_maker),
            "entity": SQLAlchemyEntityRepository(session_maker),
        }


def _create_mock_auth_client() -> AuthClient:
    """Create a mock AuthClient for testing.

    TODO: Remove this once tests are updated to use create_test_client() from nhx.testing.
    The mock is only needed because these tests use the raw FastAPI app directly
    (nhx.core.entities.api.server.app) which doesn't include AuthorizationMiddleware.
    Using create_test_client() would go through the full middleware stack and properly
    set up auth_client_context, eliminating the need for this mock.
    """
    principal = Principal(id=TEST_PRINCIPAL, email=TEST_PRINCIPAL, groups=[], on_behalf_of=None)
    config = MagicMock(spec=AuthConfig)
    config.enabled = False  # Disable auth checks in tests

    auth_client = MagicMock(spec=AuthClient)
    auth_client.principal = principal
    auth_client.auth_enabled = False
    auth_client.wait_role = AsyncMock(return_value=True)
    auth_client.has_permissions = AsyncMock(return_value=True)
    auth_client.on_behalf_of_has_permissions = AsyncMock(return_value=True)
    return auth_client


def _create_auth_client_enabled() -> AuthClient:
    """AuthClient with config.enabled True so get_accessible_workspaces enforces role bindings.

    The entities app is mounted without AuthorizationMiddleware; this matches production
    behavior for authz data scope only when :func:`nhx.common.auth.auth_client_context`
    is set per request (see ``client_with_auth``).
    """
    principal = Principal(id=TEST_PRINCIPAL, email=TEST_PRINCIPAL, groups=[], on_behalf_of=None)
    return AuthClient(
        principal=principal, config=AuthConfig(enabled=True), http_client=None, service_name="test-entities-int"
    )


def _create_auth_client_service_principal() -> AuthClient:
    """Auth enabled, principal is service:* so compute_accessible_workspaces grants all workspaces."""
    principal = Principal(id=TEST_SERVICE_PRINCIPAL, email=None, groups=[], on_behalf_of=None)
    return AuthClient(
        principal=principal,
        config=AuthConfig(enabled=True),
        http_client=None,
        service_name="test-entities-sp",
    )


def _create_auth_client_service_on_behalf_of() -> AuthClient:
    """Auth enabled: service principal with X-NHX-Principal-On-Behalf-Of to the test user (delegated scope)."""
    principal = Principal(
        id=TEST_SERVICE_PRINCIPAL,
        email=None,
        groups=[],
        on_behalf_of=TEST_PRINCIPAL,
    )
    return AuthClient(
        principal=principal,
        config=AuthConfig(enabled=True),
        http_client=None,
        service_name="test-entities-obo",
    )


class _SetAuthContextMiddleware(BaseHTTPMiddleware):
    """Mimic AuthorizationMiddleware by binding the same client used for Depends(get_auth_client)."""

    def __init__(self, app, auth_client: AuthClient) -> None:
        super().__init__(app)
        self._auth_client = auth_client

    async def dispatch(self, request, call_next):
        token = auth_client_context.set(self._auth_client)
        try:
            return await call_next(request)
        finally:
            auth_client_context.reset(token)


@pytest.fixture
async def client_with_auth(repos) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with auth enabled and context set so get_accessible_workspaces applies.

    Use with role_binding rows for the test principal; contrast with ``client`` where
    auth is off and the accessible workspace set is unrestricted.
    """
    from fastapi import FastAPI

    workspace_repo = repos["workspace"]
    entity_repo = repos["entity"]
    auth_client = _create_auth_client_enabled()

    overrides = app.dependency_overrides.copy()
    try:
        app.dependency_overrides[dep_workspace_repository_with_session] = lambda: workspace_repo
        app.dependency_overrides[dep_entity_repository_with_session] = lambda: entity_repo
        app.dependency_overrides[get_auth_client] = lambda: auth_client

        wrapper = FastAPI(title="Entities integration test wrapper (auth on)")
        wrapper.add_middleware(_SetAuthContextMiddleware, auth_client=auth_client)
        wrapper.mount("/apis/entities", app)

        transport = ASGITransport(app=wrapper)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides = overrides


@pytest.fixture
async def client_with_auth_service_principal(repos) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with auth enabled; principal is ``service:entities`` (no workspace scoping for EXISTS/child filters)."""
    from fastapi import FastAPI

    workspace_repo = repos["workspace"]
    entity_repo = repos["entity"]
    auth_client = _create_auth_client_service_principal()

    overrides = app.dependency_overrides.copy()
    try:
        app.dependency_overrides[dep_workspace_repository_with_session] = lambda: workspace_repo
        app.dependency_overrides[dep_entity_repository_with_session] = lambda: entity_repo
        app.dependency_overrides[get_auth_client] = lambda: auth_client

        wrapper = FastAPI(title="Entities integration test wrapper (service principal)")
        wrapper.add_middleware(_SetAuthContextMiddleware, auth_client=auth_client)
        wrapper.mount("/apis/entities", app)

        transport = ASGITransport(app=wrapper)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides = overrides


@pytest.fixture
async def client_with_auth_service_on_behalf_of(
    repos,
) -> AsyncGenerator[AsyncClient, None]:
    """like ``client_with_auth`` but principal is ``service:entities`` with OBO to ``TEST_PRINCIPAL``."""
    from fastapi import FastAPI

    workspace_repo = repos["workspace"]
    entity_repo = repos["entity"]
    auth_client = _create_auth_client_service_on_behalf_of()
    overrides = app.dependency_overrides.copy()
    try:
        app.dependency_overrides[dep_workspace_repository_with_session] = lambda: workspace_repo
        app.dependency_overrides[dep_entity_repository_with_session] = lambda: entity_repo
        app.dependency_overrides[get_auth_client] = lambda: auth_client

        wrapper = FastAPI(title="Entities integration test wrapper (service + on-behalf-of)")
        wrapper.add_middleware(_SetAuthContextMiddleware, auth_client=auth_client)
        wrapper.mount("/apis/entities", app)

        transport = ASGITransport(app=wrapper)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides = overrides


@pytest.fixture
async def client(repos) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing v2 API.

    Mounts the entities app at /apis/entities so that request paths match the
    platform API group (e.g. /apis/entities/v2/workspaces/...).
    """
    workspace_repo = repos["workspace"]
    entity_repo = repos["entity"]
    overrides = app.dependency_overrides.copy()
    try:
        app.dependency_overrides[dep_workspace_repository_with_session] = lambda: workspace_repo
        app.dependency_overrides[dep_entity_repository_with_session] = lambda: entity_repo
        app.dependency_overrides[get_auth_client] = _create_mock_auth_client

        # Mount entities app at /apis/entities so tests use full platform paths
        from fastapi import FastAPI

        wrapper = FastAPI(title="Entities integration test wrapper")
        wrapper.mount("/apis/entities", app)

        transport = ASGITransport(app=wrapper)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides = overrides


DEFAULT_WORKSPACE = "default"


@pytest.fixture
async def ctx(client: AsyncClient) -> Dict[str, str]:
    """Create default workspace for entity tests and return context."""
    response = await client.post(
        "/apis/entities/v2/workspaces",
        json={"name": DEFAULT_WORKSPACE, "description": "Default workspace"},
    )
    assert response.status_code == 201, f"Failed to create workspace: {response.text}"
    return {"workspace": DEFAULT_WORKSPACE}
