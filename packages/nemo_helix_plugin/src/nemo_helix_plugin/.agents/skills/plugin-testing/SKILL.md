---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-testing
description: Tests NeMo Helix plugin surfaces without a running platform. Use when writing tests for entity CRUD routes, mocking the entity client, testing NemoJob run() methods, setting up config overrides, or verifying FastAPI route error handling. Trigger keywords: test, pytest, mock entity client, TestClient, dependency_overrides, AsyncMock, test job, test config, test service, test controller.
---

# Testing NeMo Helix Plugins

All plugin surfaces can be tested without a running platform. The platform injects real implementations at runtime; tests replace them with mocks via `dependency_overrides`.

## Service Route Tests

### Setup: _make_app helper

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_my_plugin.service import MyService

def _make_app(mock_client: AsyncMock) -> FastAPI:
    service = MyService()
    app = FastAPI()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: mock_client
    return app
```

> **Critical:** Override `get_entity_client` from `nemo_helix_plugin.entity_client`, not a local stub. The routes use `Depends(get_entity_client)` — overriding a different function object has no effect and tests will raise `NotImplementedError`.

### Entity helper

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock
from nemo_helix_plugin.entity_client import NemoPaginationInfo

NOW = datetime.now(timezone.utc)

def _make_widget(name: str = "w1", workspace: str = "default") -> Widget:
    """Build a fake persisted entity with store-populated private attrs."""
    w = Widget(name=name, workspace=workspace, colour="red")
    w._id = f"id-{name}"
    w._created_at = NOW
    return w

def _make_list_response(items: list) -> MagicMock:
    resp = MagicMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=1, page_size=20, current_page_size=len(items),
        total_pages=1, total_results=len(items),
    )
    return resp
```

### Route tests (all 5 CRUD operations)

```python
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError

def test_create_201():
    mock = AsyncMock()
    mock.create.return_value = _make_widget("w1")
    client = TestClient(_make_app(mock))
    resp = client.post("/v2/workspaces/default/widgets", json={"name": "w1", "colour": "red"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "w1"

def test_create_409_conflict():
    mock = AsyncMock()
    mock.create.side_effect = NemoEntityConflictError("already exists")
    client = TestClient(_make_app(mock))
    resp = client.post("/v2/workspaces/default/widgets", json={"name": "dup", "colour": "red"})
    assert resp.status_code == 409

def test_list_200():
    mock = AsyncMock()
    mock.list.return_value = _make_list_response([_make_widget("w1"), _make_widget("w2")])
    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/widgets")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2

def test_get_200():
    mock = AsyncMock()
    mock.get.return_value = _make_widget("w1")
    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/widgets/w1")
    assert resp.status_code == 200

def test_get_404():
    mock = AsyncMock()
    mock.get.side_effect = NemoEntityNotFoundError("not found")
    client = TestClient(_make_app(mock))
    resp = client.get("/v2/workspaces/default/widgets/missing")
    assert resp.status_code == 404

def test_patch_409_version_conflict():
    mock = AsyncMock()
    mock.get.return_value = _make_widget("w1")
    mock.update.side_effect = NemoEntityConflictError("version mismatch")
    client = TestClient(_make_app(mock))
    resp = client.patch("/v2/workspaces/default/widgets/w1", json={"colour": "blue"})
    assert resp.status_code == 409

def test_delete_204():
    mock = AsyncMock()
    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/widgets/w1")
    assert resp.status_code == 204

def test_delete_404():
    mock = AsyncMock()
    mock.delete.side_effect = NemoEntityNotFoundError("not found")
    client = TestClient(_make_app(mock))
    resp = client.delete("/v2/workspaces/default/widgets/missing")
    assert resp.status_code == 404
```

## Job Tests

Jobs have no platform dependency — instantiate and call `run()` directly:

```python
from nemo_my_plugin.jobs.say_hello import SayHelloJob

def test_say_hello():
    result = SayHelloJob().run({"name": "Alice"})
    assert result == {"result": "Hello, Alice!"}

def test_say_hello_default():
    result = SayHelloJob().run({})
    assert result == {"result": "Hello, world!"}
```

> **Note:** `run()` is synchronous. Do NOT use `pytest-asyncio` for job tests — just call `job.run(config)` directly.

## Config Tests

```python
import pytest
from nemo_helix_plugin.config import set_nemo_config_override, clear_nemo_config_overrides
from nemo_my_plugin.config import MyPluginConfig

# Recommended: autouse fixture clears all overrides after every test
@pytest.fixture(autouse=True)
def reset_config():
    yield
    clear_nemo_config_overrides()

def test_debug_mode():
    set_nemo_config_override(MyPluginConfig(debug=True))
    config = MyPluginConfig.get()
    assert config.debug is True

def test_default_config():
    config = MyPluginConfig.get()
    assert config.debug is False
```

> **Always use the autouse fixture.** Config is cached per process — a test override left in place will bleed into subsequent tests.

## Discovery Cache (entry-point tests)

Tests that mock entry-points must clear the discovery cache. This is mandatory — stale cache causes test interference:

```python
import pytest
from nemo_helix_plugin.discovery import discover, discover_entry_points, discover_manifests

@pytest.fixture(autouse=True)
def clear_discovery_cache():
    yield
    discover.cache_clear()
    discover_entry_points.cache_clear()
    discover_manifests.cache_clear()
```

## Controller Tests

Controllers require no platform for unit tests — mock the entity client:

```python
import pytest
from unittest.mock import AsyncMock, patch
from nemo_my_plugin.controller import MyController

@pytest.mark.asyncio
async def test_controller_list_objects_returns_empty_on_error():
    controller = MyController()
    controller._entities = AsyncMock()
    controller._entities.list.side_effect = Exception("connection refused")
    result = await controller.list_objects()
    assert result == []
```

## Gotchas

- **Override `get_entity_client` from `nemo_helix_plugin.entity_client`**: Not a local stub. The route uses `Depends(get_entity_client)` — overriding the wrong object silently does nothing.
- **`entity.id` is `""` before persistence**: Use `_make_entity()` helpers that set `_id` directly; otherwise response validation fails.
- **`asyncio.run()` in jobs**: `run()` is sync. Tests that call `job.run()` don't need `@pytest.mark.asyncio`.
- **Config cache bleeds between tests**: Always use the `clear_nemo_config_overrides()` autouse fixture.
- **Discovery cache bleeds between tests**: Always clear `discover`, `discover_entry_points`, and `discover_manifests` caches in tests that mock entry-points.
- **`MagicMock` for list response**: Use `MagicMock()` with `.data` and `.pagination` set manually; do not import `ListResponse` from internal `nhx.common` paths.
