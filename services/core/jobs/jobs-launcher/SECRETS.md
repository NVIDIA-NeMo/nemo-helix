<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secrets Injection Feature

The jobs-launcher now supports automatic secret injection from the NeMo Helix Secrets API into subprocess environment variables.

## Configuration

The feature uses these environment variables:

1. **`NEMO_JOB_SECRETS`** (required) - A comma-separated list of secret references, e.g  `NEMO_JOB_SECRETS='ENV_VAR=workspace/secret_name,OTHER_ENV_VAR=workspace/other_secret_name'`
2. **`NHX_SECRETS_URL`** - The Secrets API endpoint (e.g., `http://localhost:8080`). Falls back to `NHX_BASE_URL` when unset; also supports `unix:///absolute/socket/path`.
3. **`NHX_PRINCIPAL`** (legacy mode only) - JSON containing the job creator's identity, e.g. `{"id":"user@example.com","email":"user@example.com","groups":["team-a"]}`.
4. **`NHX_WORKLOAD_IDENTITY_TOKEN_FILE`** (workload-identity mode) - Path to the controller-provided workload proof. The launcher exchanges this proof for a bearer token using auth discovery at `NHX_BASE_URL`, which is required in this mode.

When a workload token file is configured, Secret requests use the exchanged bearer token, with caching and refresh between requests. Delegated identity comes from the verified token claims, not `NHX_PRINCIPAL` or legacy identity headers. Discovery or token exchange failures stop the launcher without falling back to legacy headers. Authenticated Secret requests do not follow redirects.

Without a workload token file, the launcher retains the legacy `service:jobs` and on-behalf-of headers derived from `NHX_PRINCIPAL`. These headers alone cannot authenticate requests through a bearer-authenticated gateway.

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

- Secret references are configured but neither `NHX_SECRETS_URL` nor `NHX_BASE_URL` is set
- Workload authentication is configured but discovery, proof reading, or token exchange fails
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
