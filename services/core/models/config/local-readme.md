<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local dev
In this instance, running locally means running models + IGW via `uv run`, rather than via docker.

## Setup deps
```bash
cd nhx
docker compose --env-file services/core/infrastructure/models/config/local.env \
  -f deploy/quickstart/external/docker-compose.yaml \
  -f services/core/infrastructure/models/config/local-compose.yaml \
  up nhx-core
```
This will spin up all the deps of `nhx-core`, but then will make the actual `nhx-core` docker container exit.
This is done purposefully so we can then run the server ourselves.

## Run the server

```bash
export ENVFILE="services/core/models/config/local.env" && \
  uv run --frozen --env-file "$ENVFILE" nemo-helix run --services entities models inference-gateway --controllers models
```
