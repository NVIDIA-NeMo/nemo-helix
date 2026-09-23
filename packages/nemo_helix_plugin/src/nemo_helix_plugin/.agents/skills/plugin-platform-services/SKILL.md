---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-platform-services
description: Calls NeMo Helix services (entity store, jobs, files, secrets, models, inference gateway, auth) from a plugin. Use when a plugin needs to submit jobs, access files, read secrets, look up models, call the inference gateway, check permissions, or route calls between services. Trigger keywords: jobs service, files service, secrets service, models service, inference gateway, auth client, NeMo SDK, platform SDK, service-to-service, inter-service call, add_job_routes, job_route_factory, NHX_BASE_URL.
---

# Platform Services for Plugins

## SDK Access Patterns

**In request scope (FastAPI endpoint):**

```python
from nhx.common.service.dependencies import get_sdk_client
from nemo_helix import AsyncNeMoHelix
from fastapi import Depends

@router.get("/items")
async def list_items(
    workspace: str,
    sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> ...:
    models = await sdk.models.list(workspace=workspace)
    filesets = await sdk.files.list(workspace=workspace)
```

`get_sdk_client` propagates the current user's auth headers automatically. Never set `X-NHX-Principal-Id` manually in request-scope code.

**In background/controller (no request context):**

```python
from nhx.common.sdk_factory import get_async_platform_sdk

sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
```

`internal=True` adds headers that suppress the access log flood from controller polling.

## Key Env Vars

`NHX_BASE_URL` is the single most important env var — it points the SDK at the running platform:

```bash
NHX_BASE_URL=http://localhost:8080          # all services at /apis/* on this base

# Per-service URL overrides (production Kubernetes deployments):
NHX_ENTITIES_URL=http://entities:8080
NHX_JOBS_URL=http://jobs:8080
NHX_FILES_URL=http://files:8080
NHX_SECRETS_URL=http://secrets:8080
```

When a `NHX_<SERVICE>_URL` is set, the SDK factory routes that service's calls through that URL instead of the base URL.

## Entity Store (quick reference)

See `../plugin-entities/SKILL.md` for full CRUD patterns.

```python
# Request scope: use get_entity_client dependency
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client

# Background/controller scope: build manually
from nhx.common.sdk_factory import get_async_platform_sdk
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.entities.client import EntityClient

sdk = get_async_platform_sdk(as_service="my-plugin", internal=True)
entity_client = EntityClient(client_from_platform(sdk, AsyncEntitiesClient))
```

## Jobs Service

**Preferred: `add_job_routes(JobClass)`** — the plugin service mounts its job in one line. The wrapper derives everything from the `NemoJob` subclass and generates all 10 standard routes under the job collection path (POST/GET /jobs/{job-name}, GET/DELETE /jobs/{job-name}/{name}, POST /jobs/{job-name}/{name}/cancel, GET /jobs/{job-name}/{name}/logs, GET /jobs/{job-name}/{name}/results, …).

```python
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_my_plugin.jobs.process import ProcessJob  # your NemoJob subclass

router = add_job_routes(ProcessJob)
app.include_router(
    router,
    prefix="/v2/workspaces/{workspace}",
)
```

The `NemoJob` subclass declares `spec_schema` (Pydantic) and overrides `compile()` to produce a `HelixJobSpec`. See the `plugin-job` skill for the full pattern.

**Manual SDK call** (when not using `add_job_routes`):

```python
# ALWAYS pass source=service_name — without it, your jobs are invisible in list_jobs
job = await sdk.jobs.create(
    source="my-plugin",       # ← REQUIRED
    spec=job_spec,
    platform_spec=platform_spec,
    workspace=workspace,
)
status = await sdk.jobs.get_status(name=job.name, workspace=workspace)
await sdk.jobs.cancel(name=job.name, workspace=workspace)
```

## Files Service

```python
sdk: AsyncNeMoHelix = ...  # from get_sdk_client or get_async_platform_sdk

# Create a fileset
fileset = await sdk.files.create(workspace=workspace, name="my-outputs")

# List filesets
filesets = await sdk.files.list(workspace=workspace)

# Upload a file (raw HTTP — SDK doesn't wrap this)
import httpx
with open("result.json", "rb") as f:
    httpx.put(
        f"{base_url}/apis/files/v2/workspaces/{workspace}/filesets/my-outputs/-/result.json",
        content=f.read(),
    headers={"X-NHX-Principal-Id": "service:my-plugin"},  # background/controller scope only — use get_sdk_client() in request-scope FastAPI routes
    )
```

Storage backend types: `local`, `s3`, `ngc`, `huggingface` — configured via `StorageConfig` from `nhx.common.files.storage_config`.

## Secrets Service

```python
sdk: AsyncNeMoHelix = ...

# Create a secret
await sdk.secrets.create("my-api-key", workspace=workspace, value="sk-...")

# Access a secret value — POST to /access, NOT a simple GET
# Returns HelixSecretAccessResponse — the value is in .data
response = await sdk.secrets.access("my-api-key", workspace=workspace)
secret_value = response.data

# SecretRef format for storage configs
from nhx.common.api.common import SecretRef
ref = SecretRef("workspace-name/my-secret")   # workspace/name
ref = SecretRef("my-secret")                   # name only (uses request workspace)
```

`sdk.secrets.access()` calls `POST /secrets/{name}/access` internally. This is intentional — access is audited. Do NOT try to read the value via a GET.

## Models Service

```python
sdk: AsyncNeMoHelix = ...

# List available models
models = await sdk.models.list(workspace=workspace)

# Get a specific model
model = await sdk.models.retrieve(name="llama-3-8b", workspace=workspace)
# model.files_url — fileset URL for model weights
```

## Inference Gateway

OpenAI-compatible URL pattern:

```python
import httpx

# Call via OpenAI-compatible interface
resp = httpx.post(
    f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
    json={
        "model": "llama-3-8b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False,
    },
    headers={"X-NHX-Principal-Id": "service:my-plugin"},
)
resp.raise_for_status()
result = resp.json()

# Route to specific deployed model
resp = httpx.post(
    f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/model/my-model/-/v1/completions",
    json={...},
)
```

Streaming (SSE) is supported — use `stream=True` on the httpx request and iterate `resp.iter_lines()`.

## Auth

```python
from nhx.common.auth.dependencies import get_auth_client
from nhx.common.auth.client import AuthClient
from fastapi import Depends

@router.get("/items")
async def list_items(
    workspace: str,
    auth_client: AuthClient = Depends(get_auth_client),
) -> ...:
    principal_id = auth_client.principal.id
    await auth_client.authorize_request("GET", f"/apis/my-plugin/v2/workspaces/{workspace}/items")
```

The `get_auth_client` dependency is injected automatically by the platform's middleware — no setup required in plugins.

## Auth in Endpoints

```python
from nhx.common.auth.dependencies import get_auth_client
from nhx.common.auth.client import AuthClient
from fastapi import Depends

@router.get("/items")
async def list_items(
    workspace: str,
    auth_client: AuthClient = Depends(get_auth_client),
) -> ...:
    principal_id = auth_client.principal.id
    # authorize_request raises HTTPException(403) if the principal lacks permission
    await auth_client.authorize_request("GET", f"/apis/my-plugin/v2/workspaces/{workspace}/items")
```

## Job Routes

See the **Jobs Service** section above — `add_job_routes(JobClass)` from `nemo_helix_plugin.jobs.routes` is the canonical wrapper. It derives every argument from the `NemoJob` subclass and generates all 10 standard routes (create, list, get, status, delete, cancel, logs, results, get-result, download-result).

> **Always pass `source=service_name`** when creating jobs manually (jobs become invisible in the UI without it).

## See Also

- [`services-reference.md`](services-reference.md) — base URLs and key endpoints table for all services

## Gotchas

- **`source=service_name` required when creating jobs manually**: Without it, `list_jobs` for your service returns jobs from ALL services. Jobs become effectively invisible.
- **`sdk.secrets.access()` not `.get()`**: The value endpoint is `POST /access`, not `GET /{name}`. `.get()` only returns metadata (no value).
- **`internal=True` required for background/controller SDK calls**: Without it, every controller poll floods the entity store access log.
- **Never set `X-NHX-Principal-Id` manually in request-scope code**: `get_sdk_client` propagates the current user's headers automatically. Manual headers will either be ignored or cause auth failures.
- **`NHX_BASE_URL` defaults to `http://localhost:8080`**: In production this must be set to the actual cluster URL. Missing this env var is the most common cause of "connection refused" errors.
