<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secrets Injection Feature

The jobs-launcher now supports automatic secret injection from the NeMo Helix Secrets API into subprocess environment variables.

## Configuration

The feature uses these environment variables:

1. **`NEMO_JOB_SECRETS`** (required) - A comma-separated list of secret references, e.g  `NEMO_JOB_SECRETS='ENV_VAR=workspace/secret_name,OTHER_ENV_VAR=workspace/other_secret_name'`
2. **`NHX_SECRETS_URL`** - The Secrets API base URL. Falls back to `NHX_BASE_URL`.
3. **`NHX_WORKLOAD_IDENTITY_TOKEN_FILE`** - When set, the launcher discovers workload exchange settings through `NHX_BASE_URL`, exchanges this projected token, and sends a refreshable bearer token for secret retrieval. The same authentication supports `--fetch-step-config` and log uploads. Exchange failures stop the launch; they never fall back to trusted headers or anonymous secret requests.
4. **`NHX_PRINCIPAL`** - Legacy JSON auth context used only when workload identity is not configured, e.g. `{"id":"user@example.com","email":"user@example.com","groups":["team-a"]}`. Workload mode needs no principal environment variable and removes legacy identity headers from bearer-authenticated requests.

With workload exchange, the Kubernetes job service account needs the `SecretAccessor` role in
the secret's platform workspace. This grants `secrets.access`; the delegated submitter must
separately have `secrets.read`. The API accepts this path only for delegated job-bound Kubernetes
workloads or service principals. `JobRunner` alone permits log upload, not secret access.
Operators must create the role binding explicitly; Kubernetes RBAC does not grant Helix roles.

The launcher uses configured workload credentials, an access-token file, or the creator in `NHX_PRINCIPAL`. If none is configured, it sends an anonymous request without auth discovery; the Secrets API enforces the platform's authentication and permission settings. Anonymous requests support internal HTTP service URLs. URL-embedded credentials are rejected for anonymous Secret requests. A configured credential that fails is an error, and a failed Secret request prevents the subprocess from starting.

## Usage Example

```bash
# Set the required environment variables
export NHX_SECRETS_URL="http://localhost:8080"
export NEMO_JOB_SECRETS="HF_TOKEN=default/hf-token,WANDB_API_KEY=default/wandb-key"
export NHX_PRINCIPAL='{"id":"user@example.com"}'

# Run your command - secrets will be injected as environment variables
jobs-launcher run python train.py
```

## How It Works

1. The launcher parses the `NEMO_JOB_SECRETS` environment variable
2. For each secret reference (e.g., `HF_TOKEN=default/hf-token`):
   - Extracts the environment variable name (`HF_TOKEN`) and secret reference (`default/hf-token`)
   - Calls the NeMo Helix API: `GET /apis/secrets/v2/workspaces/default/secrets/hf-token/access`
   - Retrieves the secret data
   - Creates an environment variable: `HF_TOKEN=<secret-value>`
3. The subprocess is launched with all inherited environment variables plus the fetched secrets

## Secret Reference Format

Each secret reference must follow the format: `ENV_VAR=workspace/secret_name`

The environment variable name (left side of `=`) is what gets set in the subprocess environment.
The workspace/secret_name (right side of `=`) is what gets fetched from the API.

Examples:

- `HF_TOKEN=default/hf-token` - Fetches secret "hf-token" from "default" workspace and sets it as HF_TOKEN
- `DATABASE_PASSWORD=production/db-pass` - Fetches secret "db-pass" from "production" workspace and sets it as DATABASE_PASSWORD

Multiple secrets are separated by commas:

```bash
export NEMO_JOB_SECRETS="HF_TOKEN=default/hf-token,WANDB_API_KEY=default/wandb-key,DB_PASS=production/db-password"
```

## Error Handling

The launcher will exit with an error if:

- `NEMO_JOB_SECRETS` is set but both `NHX_SECRETS_URL` and `NHX_BASE_URL` are missing
- A secret reference has an invalid format
- The API returns an error (e.g., secret not found, authentication failure)
- Network connectivity issues occur

## Logging

The launcher logs secret retrieval operations:

```
[launcher] Fetching secret HF_TOKEN from workspace default...
[launcher] Successfully fetched secret HF_TOKEN
[launcher] Injected 2 secret(s) as environment variables
```

Note: The actual secret values are never logged.
