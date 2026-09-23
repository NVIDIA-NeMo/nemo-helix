<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix SDK

## Introduction

The Python SDK for NeMo Helix is checked in to this repository under `sdk/python/nemo-helix` and is maintained directly here. The [OpenAPI specification](../openapi/openapi.yaml) documents the platform's HTTP API. The SDK provides synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx), with type safety, a well-defined error hierarchy, configurable retries, timeouts, streaming, raw response access, and logging integration.

The SDK package is not regenerated. If a previously generated type or client needs to change, do not edit it here: use the corresponding typed client from `nemo_helix_plugin` instead and migrate consumers to it.

## Folder Structure

- `sdk/python/nemo-helix` — the hand-maintained Python SDK.
- The OpenAPI spec at `openapi/openapi.yaml` is the source of truth for the platform's HTTP API routes.

## Updating the OpenAPI spec

The OpenAPI spec is regenerated locally from the FastAPI service code (no external credentials required):

```shell
make refresh-openapi
```

This updates `openapi/openapi.yaml`. The OpenAPI generation also runs as a pre-commit hook (manual stage) when API files change.

## SDK maintenance

SDK maintenance commands are provided by `nemo-helix-sdk-tools`. See `uv run nemo-helix-sdk-tools --help` for details.
