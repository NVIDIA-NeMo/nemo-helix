<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Model Configuration (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Your task is to configure model providers for use with Data Designer.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Configure the inference infrastructure needed for Data Designer by completing these operations:

1. Create a secret to store an API key for the model provider
2. Create a model provider that references this secret
3. Verify the model provider was created and is ready for use

## Details

Use these specific values:
- **Secret name**: `dd-test-api-key`
- **Secret value**: `test-key-12345` (this is a test environment)
- **Provider name**: `dd-test-provider`
- **Provider host URL**: `https://integrate.api.nvidia.com`
- **Provider description**: "Test provider for Data Designer evaluation"

## Success Criteria

The task is complete when:
- The secret `dd-test-api-key` has been created
- The model provider `dd-test-provider` has been created with the correct host URL
- The provider references the secret `dd-test-api-key`
