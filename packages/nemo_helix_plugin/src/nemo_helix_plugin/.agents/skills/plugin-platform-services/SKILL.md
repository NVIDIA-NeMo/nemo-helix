---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-platform-services
description: Calls NeMo Helix services (entity store, jobs, files, secrets, models, inference gateway, auth) from a plugin. Use when a plugin needs to submit jobs, access files, read secrets, look up models, call the inference gateway, check permissions, or route calls between services. Trigger keywords: jobs service, files service, secrets service, models service, inference gateway, auth client, typed client, NemoClient, platform client, service-to-service, inter-service call, add_job_routes, job_route_factory, NHX_BASE_URL.
---

# NeMo Helix Services for Plugins

## Typed Client Access Patterns

Every platform call goes through a typed client from `nemo_helix_plugin`: a base
`AsyncNemoClient` / `NemoClient`, plus per-service clients derived from it with
`<ServiceClient>.from_client(client)` (they share the base client's transport, auth and headers).

**In request scope (FastAPI endpoint):**

```python
from fastapi import Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.models.client import AsyncModelsClient

@router.get("/items")
async def list_items(
    workspace: str,
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> ...:
    models = await AsyncModelsClient.from_client(client).list_models(workspace=workspace)
    filesets = await AsyncFilesClient.from_client(client).list_filesets(workspace=workspace)
    async for model in models.items():
        ...
```

`get_nemo_client` hands out a request-scoped client that propagates the current user's auth headers automatically. Never set `X-NHX-Principal-Id` manually in request-scope code. Use `get_sync_nemo_client` for sync-only code paths.

**In background/controller (no request context):**

```python
from nemo_helix_plugin.client_provider import get_async_nemo_client

client = get_async_nemo_client(as_service="my-plugin", internal=True)
```

`internal=True` adds headers that suppress the access log flood from controller polling.

**Acting for a user from background code** (authorize as the entity owner, not the service principal):

```python
client = get_async_nemo_client(as_service="my-plugin", internal=True, on_behalf_of=owner_principal_id)
```

**Inside a task container** (authenticates as the service, delegated to the job creator from `NHX_PRINCIPAL`):

```python
from nemo_helix_plugin.client_provider import get_task_nemo_client

client = get_task_nemo_client("my-plugin")
```

## Key Env Vars

`NHX_BASE_URL` is the single most important env var — it points platform clients at the running platform:

```bash
NHX_BASE_URL=http://localhost:8080          # all services at /apis/* on this base

# Per-service URL overrides (production Kubernetes deployments):
NHX_ENTITIES_URL=http://entities:8080
NHX_JOBS_URL=http://jobs:8080
NHX_FILES_URL=http://files:8080
NHX_SECRETS_URL=http://secrets:8080
```

When a `NHX_<SERVICE>_URL` is set, the platform client provider routes that service's calls through that URL instead of the base URL.

## Entity Store (quick reference)

See `../plugin-entities/SKILL.md` for full CRUD patterns.

```python
# Request scope: use get_entity_client dependency
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client

# Background/controller scope: build manually
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.entities.client import AsyncEntitiesClient

client = get_async_nemo_client(as_service="my-plugin", internal=True)
entity_client = NemoEntitiesClient(AsyncEntitiesClient.from_client(client))
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

**Manual typed-client call** (when not using `add_job_routes`):

```python
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

jobs_client = AsyncJobsClient.from_client(client)
# ALWAYS pass source=service_name — without it, your jobs are invisible in list_jobs
job = await jobs_client.create_job(
    workspace=workspace,
    body=CreateHelixJobRequest(
        source="my-plugin",       # ← REQUIRED
        spec=job_spec,
        platform_spec=platform_spec,
    ),
)
status = await jobs_client.get_job_status(workspace=workspace, name=job.name)
await jobs_client.cancel_job(workspace=workspace, name=job.name)
```

## Files Service

```python
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest

files_client = AsyncFilesClient.from_client(client)  # client from get_nemo_client or get_async_nemo_client

# Create a fileset (exist_ok returns the existing fileset on 409)
fileset = await files_client.create_fileset(
    workspace=workspace, body=CreateFilesetRequest(name="my-outputs"), exist_ok=True
)

# List filesets
async for fs in (await files_client.list_filesets(workspace=workspace)).items():
    ...

# Upload / download a file
with open("result.json", "rb") as f:
    await files_client.upload_file(workspace=workspace, name="my-outputs", path="result.json", content=f.read())
data = await (await files_client.download_file(workspace=workspace, name="my-outputs", path="result.json")).read()
```

Storage backend types: `local`, `s3`, `ngc`, `huggingface` — configured via `StorageConfig` from `nhx.common.files.storage_config`.

## Secrets Service

```python
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest

secrets_client = AsyncSecretsClient.from_client(client)

# Create a secret
await secrets_client.create_secret(
    workspace=workspace, body=HelixSecretCreateRequest(name="my-api-key", value="sk-...")
)

# Access a secret value — POST to /access, NOT a simple GET
# Returns HelixSecretAccessResponse — the value is in .data
response = await secrets_client.access_secret(workspace=workspace, name="my-api-key")
secret_value = response.data

# SecretRef format for storage configs
from nhx.common.api.common import SecretRef
ref = SecretRef("workspace-name/my-secret")   # workspace/name
ref = SecretRef("my-secret")                   # name only (uses request workspace)
```

`access_secret()` calls `POST /secrets/{name}/access`. This is intentional — access is audited. Do NOT try to read the value via a GET.

## Models Service

```python
from nemo_helix_plugin.models.client import AsyncModelsClient

models_client = AsyncModelsClient.from_client(client)

# List available models (paginated; .items() walks every page)
async for model in (await models_client.list_models(workspace=workspace)).items():
    ...

# Get a specific model
model = await models_client.get_model(workspace=workspace, name="llama-3-8b")
# model.files_url — fileset URL for model weights
```

Typed clients raise `nemo_helix_plugin.client.errors` exceptions: `NotFoundError`, `ConflictError`, … (all `NemoHTTPError`, with `.status_code` and `.detail`) and `NemoTransportError` for connection failures.

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
- **`access_secret()` not `get_secret()`**: The value endpoint is `POST /access`, not `GET /{name}`. `get_secret()` only returns metadata (no value).
- **`internal=True` required for background/controller clients**: Without it, every controller poll floods the entity store access log.
- **Never set `X-NHX-Principal-Id` manually in request-scope code**: `get_nemo_client` propagates the current user's headers automatically. Manual headers will either be ignored or cause auth failures.
- **`NHX_BASE_URL` defaults to `http://localhost:8080`**: In production this must be set to the actual cluster URL. Missing this env var is the most common cause of "connection refused" errors.
