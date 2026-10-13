<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authentik Docker Compose Runtime

This directory contains the Docker Compose runtime files for the Authentik
reference example.

Use the single shared tutorial for the end-to-end walkthrough:

- [Authentik Reference Tutorial](../tutorial.md)

For Compose-specific architecture and wiring, see:

- [Implementation Details](implementation-details.md)

From the repo root, start the Compose runtime with:

```bash
contrib/auth/authentik/run.sh up compose
```

The harness prepares the generated local secrets, starts Compose, waits for the
gateway, and registers the `authentik-compose` NeMo CLI context. Stop it with:

```bash
contrib/auth/authentik/run.sh down compose
```

The harness Compose project name defaults to `authentik-e2e-reuse`, so
container, network, and volume names do not inherit the generic `compose`
directory name. Set `NHX_AUTHENTIK_COMPOSE_PROJECT_NAME` before running the
harness if you need a different local namespace.
