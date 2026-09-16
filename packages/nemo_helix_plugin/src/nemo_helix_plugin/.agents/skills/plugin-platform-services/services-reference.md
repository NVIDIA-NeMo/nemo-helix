<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Platform Services Quick Reference

**Contents:** [Entity Store](#entity-store) · [Jobs Service](#jobs-service) · [Files Service](#files-service-filesets) · [Secrets Service](#secrets-service) · [Models Service](#models-service) · [Inference Gateway](#inference-gateway) · [Auth Service](#auth-service)

---

## Entity Store

**Base URL:** `/apis/entities/v2`

| Method | Path | Description |
|---|---|---|
| `POST` | `/workspaces/{workspace}/entities/{entity_type}` | Create entity |
| `GET` | `/workspaces/{workspace}/entities/{entity_type}` | List entities (filter/sort/page) |
| `GET` | `/workspaces/{workspace}/entities/{entity_type}/{name}` | Get by name |
| `PATCH` | `/workspaces/{workspace}/entities/{entity_type}/{name}` | Update (optimistic lock) |
| `DELETE` | `/workspaces/{workspace}/entities/{entity_type}/{name}` | Delete |
| `GET` | `/entities/{id}` | Get by UUID |
| `GET` | `/workspaces` | List workspaces |
| `POST` | `/workspaces` | Create workspace |
| `GET` | `/workspaces/{workspace}/members` | List workspace members |
| `POST` | `/workspaces/{workspace}/members` | Add workspace member |

**Env vars:**
```bash
NHX_BASE_URL=http://localhost:8080
NHX_ENTITIES_URL=http://entities:8080   # optional direct URL
```

**Auth:** `X-NHX-Principal-Id` required. Use `service:my-plugin` for controllers.

---

## Jobs Service

**Base URL:** `/apis/jobs/v2`

| Method | Path | Description |
|---|---|---|
| `POST` | `/workspaces/{workspace}/jobs` | Submit job (`source` field required) |
| `GET` | `/workspaces/{workspace}/jobs` | List jobs (filter/search/sort) |
| `GET` | `/workspaces/{workspace}/jobs/{name}` | Get job |
| `GET` | `/workspaces/{workspace}/jobs/{name}/status` | Get job status |
| `DELETE` | `/workspaces/{workspace}/jobs/{name}` | Delete job |
| `POST` | `/workspaces/{workspace}/jobs/{name}/cancel` | Cancel job |
| `POST` | `/workspaces/{workspace}/jobs/{name}/pause` | Pause job |
| `POST` | `/workspaces/{workspace}/jobs/{name}/resume` | Resume job |
| `GET` | `/workspaces/{workspace}/jobs/{name}/logs` | Get logs |
| `GET` | `/workspaces/{workspace}/jobs/{name}/results` | List results |
| `GET` | `/workspaces/{workspace}/jobs/{job}/results/{name}` | Get result |
| `GET` | `/workspaces/{workspace}/jobs/{job}/results/{name}/download` | Download result |
| `GET` | `/execution-profiles` | List hardware profiles |

**Job status states:** `created → pending → active → completed` (or `error`, `cancelled`, `paused`)

**Env vars:**
```bash
NHX_BASE_URL=http://localhost:8080
NHX_JOBS_URL=http://jobs:8080         # optional direct URL
```

**Auth:** `X-NHX-Principal-Id` required. **Always pass `source=service_name`** when creating jobs or list results will be mixed with other services.

---

## Files Service (Filesets)

**Base URL:** `/apis/files/v2`

| Method | Path | Description |
|---|---|---|
| `POST` | `/workspaces/{workspace}/filesets` | Create fileset |
| `GET` | `/workspaces/{workspace}/filesets` | List filesets |
| `GET` | `/workspaces/{workspace}/filesets/{name}` | Get fileset |
| `PATCH` | `/workspaces/{workspace}/filesets/{name}` | Update fileset |
| `DELETE` | `/workspaces/{workspace}/filesets/{name}` | Delete fileset |
| `GET` | `/workspaces/{workspace}/filesets/{name}/files` | List files in fileset |
| `GET` | `/workspaces/{workspace}/filesets/{name}/-/{path}` | Download file |
| `PUT` | `/workspaces/{workspace}/filesets/{name}/-/{path}` | Upload file |
| `DELETE` | `/workspaces/{workspace}/filesets/{name}/-/{path}` | Delete file |

**Storage backends:** `local`, `s3`, `ngc`, `huggingface`

**Env vars:**
```bash
NHX_BASE_URL=http://localhost:8080
NHX_FILES_URL=http://files:8080       # optional direct URL
```

---

## Secrets Service

**Base URL:** `/apis/secrets/v2`

| Method | Path | Description |
|---|---|---|
| `POST` | `/workspaces/{workspace}/secrets` | Create secret |
| `GET` | `/workspaces/{workspace}/secrets` | List secrets (names only, no values) |
| `GET` | `/workspaces/{workspace}/secrets/{name}` | Get metadata (no value) |
| `PATCH` | `/workspaces/{workspace}/secrets/{name}` | Update secret value |
| `DELETE` | `/workspaces/{workspace}/secrets/{name}` | Delete secret |
| `POST` | `/workspaces/{workspace}/secrets/{name}/access` | **Access secret value** |
| `POST` | `/rotate-encryption-keys` | Rotate all encryption keys |

**Auth:** Tightly controlled. `X-NHX-Principal-On-Behalf-Of` for acting on behalf of a user.

**Env vars:**
```bash
NHX_BASE_URL=http://localhost:8080
NHX_SECRETS_URL=http://secrets:8080   # optional direct URL
```

---

## Models Service

**Base URL:** `/apis/models/v2`

| Method | Path | Description |
|---|---|---|
| `POST` | `/workspaces/{workspace}/models` | Register model |
| `GET` | `/workspaces/{workspace}/models` | List models |
| `GET` | `/workspaces/{workspace}/models/{name}` | Get model |
| `PATCH` | `/workspaces/{workspace}/models/{name}` | Update model |
| `DELETE` | `/workspaces/{workspace}/models/{name}` | Delete model |
| `GET` | `/workspaces/{workspace}/models/{model}/adapters` | List adapters |
| `POST` | `/workspaces/{workspace}/models/{model}/adapters` | Add adapter |
| `POST` | `/workspaces/{workspace}/deployments` | Deploy model |
| `GET` | `/workspaces/{workspace}/deployments` | List deployments |
| `GET` | `/workspaces/{workspace}/deployments/{name}` | Get deployment |
| `DELETE` | `/workspaces/{workspace}/deployments/{name}` | Undeploy |
| `GET` | `/workspaces/{workspace}/deployments/{name}/status` | Deployment status |
| `POST` | `/workspaces/{workspace}/providers` | Register provider |
| `GET` | `/workspaces/{workspace}/providers` | List providers |
| `GET` | `/workspaces/{workspace}/providers/{name}/status` | Provider readiness |

**Env vars:**
```bash
NHX_BASE_URL=http://localhost:8080
NHX_MODELS_URL=http://models:8080     # optional direct URL
```

---

## Inference Gateway

**Base URL:** `/apis/inference-gateway/v2`

| Method | Path | Description |
|---|---|---|
| `GET` | `/workspaces/{workspace}/openai/-/v1/models` | List available models |
| `POST` | `/workspaces/{workspace}/openai/-/{trailing_uri}` | OpenAI-compatible forward |
| `POST` | `/workspaces/{workspace}/model/{name}/-/{trailing_uri}` | Route to specific model |
| `POST` | `/workspaces/{workspace}/provider/{name}/-/{trailing_uri}` | Route to specific provider |
| `GET` | `/workspaces/{workspace}/provider/{name}/ready` | Check provider readiness |

Supports SSE streaming. `trailing_uri` captures the full sub-path (e.g., `v1/chat/completions`).

**Auth:** `X-NHX-Principal-Id` required. Gateway checks workspace membership.

---

## Auth Service

**Base URL:** `/apis/auth`

| Method | Path | Description |
|---|---|---|
| `POST` | `/v2/iam/role-bindings` | Create role binding |
| `GET` | `/v2/iam/role-bindings` | List role bindings |
| `GET` | `/v2/iam/role-bindings/{name}` | Get role binding |
| `DELETE` | `/v2/iam/role-bindings/{name}` | Delete role binding |
| `GET` | `/discovery` | OIDC discovery document |

**In-process usage via FastAPI dependency:**
```python
from nhx.common.auth.dependencies import get_auth_client
from nhx.common.auth.client import AuthClient

auth_client: AuthClient = Depends(get_auth_client)
principal_id = auth_client.principal.id
await auth_client.authorize_request("GET", f"/apis/my-plugin/...")
```

**Auth config in `/etc/nhx/config.yaml`:**
```yaml
auth:
  enabled: true
  policy_decision_point_base_url: http://localhost:8080
  policy_decision_point_provider: embedded  # or opa
```
