<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix SDK

> **Using SDK generation for the first time? Start with [setup instructions](./setup-instructions.md)**.

## Introduction

We are using [Stainless](https://www.stainless.com/) to generate a Python SDK for NeMo Helix from our OpenAPI specification. Stainless provides high-quality, idiomatic Python clients with comprehensive features, including synchronous and asynchronous interfaces, type safety, error handling, and more.

The generated SDK includes:

1. **Sync and Async Clients**: Both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx)
2. **Type Safety**: Comprehensive type definitions for all request parameters and response fields
3. **Error Handling**: Well-defined exception hierarchy with context-specific error types
4. **Configurable Retries**: Automatic retries with exponential backoff for transient errors
5. **Timeout Configuration**: Flexible timeout settings for different types of operations
6. **Streaming Support**: Ability to stream responses for large content
7. **Raw Response Access**: Access to underlying HTTP response data when needed
8. **Logging Integration**: Integration with Python's standard logging module

## Links

The Stainless Webapp is available at: https://app.stainless.com/nvidia.

The project used for generating the Python SDK is `nemo-helix-v1-python`: https://app.stainless.com/nvidia/nemo-helix-v1/overview.

The associated GitHub repository is:
- Generated client code: https://github.com/stainless-sdks/nemo-helix-v1-python

## Folder Structure

The SDK inside the Platform repo is structured as follows:

- The `sdk` folder contains all the files related to the SDK.
  - The `sdk/stainless.yaml` file contains the configuration for Stainless.
  - The `sdk/python/nemo-helix` folder contains the generated Python SDK.

## Updating the SDK

> **This section assumes that steps from [setup instructions](./setup-instructions.md) have been completed**.

At a high-level, the SDK update process:
- Takes 2 input files:
  - The OpenAPI spec at `openapi/openapi.yaml`
  - The Stainless config at `sdk/stainless.yaml`
- Runs the Stainless code generation process (in Stainless' cloud).
- Fetches the updated code from the GitHub repository and updates the local copy of the SDK in the Platform repo (`sdk/python/nemo-helix`).

This whole process can be executed by running:

```shell
make stainless
```

Or directly via the script:

```shell
./sdk/stainless.sh sync
```

This will modify the local copy of the SDK in the Platform repo, then you can commit the changes to the Platform repo.

The `sync` command waits for the code generation (commit phase) to complete before pulling the changes. The Stainless Webapp link is printed so you can monitor progress.

### Pulling Updates

To pull the changes from the GitHub repository, run the following command:

```bash
./sdk/stainless.sh pull
```

### Using the Stainless Webapp

The code generation can also be triggered from Stainless Webapp directly. This is useful for debugging purposes.
1. Go to the [Stainless Webapp](https://app.stainless.com/nvidia/nemo-helix-v1/studio?language=python).
2. (Recommended) Pick your branch from dropdown in the top right corner.
3. Update the "Stainless Config" and/or "OpenAPI Spec".
4. Click the "Save" button (Cmd+S works as well) to trigger the code generation.

## EA SDK

> **EA SDK is now included in the main SDK, under `beta` prefix**

## Implementation details

Behind the scenes, the key command that triggers the updates is the [Stainless CLI](https://github.com/stainless-api/stainless-api-cli):

```bash
stl builds create \
    --project nemo-helix \
    --branch <current-branch> \
    --config sdk/stainless.yaml \
    --openapi-spec openapi/openapi.yaml \
    --wait commit \
    --allow-empty
```

Key flags:
- `--wait commit`: Wait for the code generation (commit phase) to complete, but don't wait for lint/test/build workflows
- `--allow-empty`: Don't fail if there are no changes to commit
- `--openapi-spec`: Path to the OpenAPI specification file

The CLI uses the `STAINLESS_API_KEY` environment variable for authentication.

By default, the branch name in Stainless is set to the current branch in this repository (`git rev-parse --abbrev-ref HEAD`). If you'd like to use a different branch, assign it to `STAINLESS_BRANCH` env var before running sync.

### Installing the stl CLI

The `stl` CLI is installed automatically when running `make stainless`. You can also install it manually:

```bash
# via Homebrew (macOS)
brew install stainless-api/tap/stl
```

### Troubleshooting

#### `stainless.sh sync` failed

Common reasons for failure:

1. **OpenAPI spec or Stainless config errors**: The code generation failed due to invalid input.
   - Navigate to the Stainless Webapp (link printed by the `sync` command)
   - Check for error messages in the build output
   - Fix the issues in your OpenAPI spec or Stainless config and try again

2. **Network or authentication issues**: The CLI couldn't connect to Stainless.
   - Verify `STAINLESS_API_KEY` is set correctly
   - Check your network connection

SDK maintenance commands are provided by `nemo-helix-sdk-tools`. See `uv run nemo-helix-sdk-tools --help` for details.
