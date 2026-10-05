<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Integrations setup (W&B + MLflow, local / Docker platform)

Use this when job JSON includes `integrations.wandb` and/or `integrations.mlflow` on a **local or single-node Docker** NeMo Helix (`platform.runtime: docker`) — i.e. **automodel / unsloth**. Field reference: `hyperparameters.md` § **Integrations (all backends)**. rl (DPO) accepts the same `integrations` block but runs on **Kubernetes / Ray**: reuse the field reference, but point `tracking_uri` / self-hosted W&B `base_url` at an endpoint reachable from the cluster (the `docker0` recipe below is Docker-runtime only).

## MLflow — local tracking server

Run MLflow on the **platform host** (the machine whose Docker daemon runs training containers):

```bash
docker run -d \
  --name mlflow \
  --restart unless-stopped \
  -p 5001:5000 \
  -v mlflow-data:/mlflow \
  ghcr.io/mlflow/mlflow:v2.18.0 \
  mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts
```

UI: `http://<platform-host>:5001`

### `tracking_uri` for training containers

Training steps run in Docker. Set `integrations.mlflow.tracking_uri` to an address the **container** can reach — not `localhost` inside the container.

1. Resolve the host bridge IP (often `docker0`):

```bash
export DOCKER_HOST_IP=$(ip -4 addr show docker0 | awk '/inet / {print $2}' | cut -d/ -f1)
echo "$DOCKER_HOST_IP"
```

1. Use that IP with the **published** host port (`5001` in the command above):

```json
"mlflow": {
  "experiment_name": "customizer-integration",
  "name": "my-run",
  "tracking_uri": "http://${DOCKER_HOST_IP}:5001"
}
```

Substitute `${DOCKER_HOST_IP}` with the value from step 1 (JSON does not expand shell variables). With `job_container_network: host` on the GPU execution profile, the host's LAN IP or `docker0` may both work — use whichever you verified from a running training container.

## jobs-launcher — required for `WANDB_API_KEY` injection

W&B (and other `from_secret` env vars) are injected by **jobs-launcher** before the training entrypoint runs. If launcher is missing or misconfigured, training starts without `WANDB_API_KEY` even when `integrations.wandb.api_key_secret` is set.

On the **platform host**, from the nemo-helix git root:

```bash
cd services/core/jobs/jobs-launcher
./build-manual.sh linux amd64
cd ../../../..
```

Point platform config at the built binary (absolute path), e.g. in `~/.nemo/config.yaml` under the Docker jobs executor:

```yaml
jobs:
  executors:
    docker:
      launcher_tool_path: /path/to/nemo-helix/services/core/jobs/jobs-launcher/jobs-launcher
```

Restart Helix services after changing launcher path:

```bash
uv run nemo services restart
```

Successful injection appears in training logs:

```text
[launcher] Successfully fetched secret wandb-api-key and mapped to WANDB_API_KEY
```

## W&B — platform secret

Job JSON references the secret by name:

```json
"wandb": {
  "project": "customizer-integration",
  "name": "my-run",
  "entity": "Nemo-automodel",
  "api_key_secret": "default/wandb-api-key"
}
```

`default/wandb-api-key` means workspace `default`, secret name `wandb-api-key`.

Store the API key in the **platform** secret store. A local `wandb login` cache on your laptop is **not** used by training containers.

```bash
export NHX_BASE_URL=http://<platform-host>:8080   # omit when using default localhost
cd /path/to/nemo-helix

# Create (first time)
uv run nemo secrets create wandb-api-key \
  --value "$WANDB_API_KEY" \
  --workspace default

# Update (replace placeholder or rotated key)
uv run nemo secrets update wandb-api-key \
  --value "$WANDB_API_KEY" \
  --workspace default
```

Prefer piping when the key has special characters:

```bash
printf '%s' "$WANDB_API_KEY" | uv run nemo secrets update wandb-api-key --from-file - --workspace default
```

Get a key from https://wandb.ai/authorize (User settings → API keys).

## Unsloth image note

`nhx-unsloth-training` must include the `[integrations]` extra (`wandb`, `mlflow-skinny`) or HF `WandbCallback` / MLflow callbacks fail at trainer init. Rebuild and set `NHX_UNSLOTH_TRAINING_IMAGE` on the platform host after Dockerfile changes.

## Verify end-to-end

1. Submit a job with both integrations (fixtures: `plugins/nemo-automodel/tests/fixtures/integrations_wandb_mlflow.json`, `plugins/nemo-unsloth/tests/fixtures/integrations_wandb_mlflow.json`).
2. Training logs: launcher secret fetch, `wandb: Syncing run …`, MLflow run under the configured experiment.
3. W&B UI: `https://wandb.ai/<entity>/<project>`
4. MLflow UI: `http://<platform-host>:5001`
