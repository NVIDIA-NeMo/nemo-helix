---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Build and start the NeMo Helix quickstart environment with docker-compose
---
Start the NeMo Helix quickstart environment with the specified services and controllers

## Workflow

Determine the command based on options, each line being one example on how to run things

```bash
  export DATABASE_DIALECT=sqlite
  export DATABASE_PATH=$HOME/.local/share/nemo/nhx-platform.db
  export UVICORN_RELOAD=true
  # Default
  uv run nemo-helix run --quickstart
  # Specific config
  uv run nemo-helix run --config packages/nhx_platform/config/local.yaml
  # One service
  uv run nemo-helix run --services hello-world
  # Jobs service
  uv run nemo-helix run --controllers jobs
  # Jobs service and controller
  uv run nemo-helix run --services jobs --controllers jobs
  uv run nemo-helix run task --task nhx.hello_world.tasks.hello_world
  uv run nemo-helix run task --task nhx.hello_world.tasks.hello_world --config '{"key": "value"}
```
