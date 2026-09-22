<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK Maintenance Tools

This package contains repo-local commands for keeping the generated Python SDK, Stainless config, generated CLI, vendored packages, and license metadata in sync.

## OpenAPI -> Stainless config mapper

OpenAPI spec to Stainless config model mapper.

This tool maps OpenAPI spec to the Stainless configuration by:
- creating Stainless methods for each OpenAPI endpoint (if not already present). It will also remove stale methods.
- creating Stainless models for each OpenAPI schema (if not already present). It will also remove stale models.

See https://www.stainless.com/docs/guides/configure#methods and https://www.stainless.com/docs/guides/configure#models

### Methods

Can be run as:

```sh
uv run --frozen nemo-platform-sdk-tools openapi-stainless sync-methods \
  --openapi-spec-path openapi/openapi.yaml \
  --stainless-config-path sdk/stainless.yaml \
  --output-path sdk/stainless.yaml
```

It will:
- Find all OpenAPI endpoints
- Compare with existing Stainless methods
- Add missing methods with "reviewme_" prefix to method name
- Remove stale methods (i.e. methods that exist in Stainless but not in OpenAPI)

**Note: because the way methods are organized in Stainless, it wouldn't be reliable to automatically determine the
method name and its location, that's why the method name is prefixed with "reviewme_" and the developer is expected to
review and update both the name and potentially the location (in best case, if both are correct, just remove the prefix).**

### Models

Can be run as:
```sh
uv run --frozen nemo-platform-sdk-tools openapi-stainless sync-models \
  --openapi-spec-path openapi/openapi.yaml \
  --stainless-config-path sdk/stainless.yaml \
  --output-path sdk/stainless.yaml
```

It will:
- Check all Stainless methods are in sync with OpenAPI endpoints. If not, run `sync-methods` first.
- Find all OpenAPI schemas
- Compare with existing Stainless models
- Add missing models with a name derived from the schema name
- **If there is a conflict, it will add the "reviewme_" prefix to the model name and the developer is expected to review and update the name**

## SDK Freshness Check

Check whether the generated SDK matches the OpenAPI spec and Stainless config:

```sh
uv run --frozen nemo-platform-sdk-tools is-up-to-date --output-dir python-sdk-lint
```

## SDK Vendoring

Vendor configured platform packages into the Python SDK wrapper:

```sh
uv run --no-sync nemo-platform-sdk-tools vendor all-from-configs \
  nemo_platform_ext models filesets nemo_evaluator_sdk
```

Run post-generation updates:

```sh
uv run --no-sync nemo-platform-sdk-tools post-generation update-license-headers
uv run --frozen nemo-platform-sdk-tools post-generation update-pyproject
uv run --frozen nemo-platform-sdk-tools post-generation update-all
```

Prefer the Makefile targets (`make vendor` and `make update-sdk`) for normal repo workflows.
