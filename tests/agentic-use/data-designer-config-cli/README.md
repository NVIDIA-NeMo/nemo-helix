<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Model Configuration - CLI Eval

This Harbor eval tests the ability to configure model providers for Data Designer using the NeMo Helix CLI.

## What This Eval Tests

Data Designer requires model providers to be configured before it can generate synthetic data. This eval tests:

1. **Secret creation** - Creating a secret to store the API key for the model provider
2. **Provider creation** - Creating a model provider with the correct host URL and API key reference
3. **Configuration verification** - Ensuring the provider is properly configured

## Flow Reference

This eval corresponds to **Flow #26 - Configure Models** from the Data Designer service flows.

## Running the Eval

1. Build the agentic-base image:
   ```bash
   docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
   ```

2. Set up your Anthropic API credentials:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key'
   export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
   ```

3. Run with Harbor:
   ```bash
   harbor run -p tests/agentic-use/data-designer-config-cli \
       --agent claude-code \
       --model aws/anthropic/bedrock-claude-sonnet-4-5-v1
   ```

## Expected Outcome

The agent should:
1. Create a secret named `dd-test-api-key` with the test API key value
2. Create a model provider named `dd-test-provider` that references the secret
3. The provider should have the correct host URL (`https://integrate.api.nvidia.com`)

## Verification

The verifier tests check:
- Secret `dd-test-api-key` exists
- Provider `dd-test-provider` exists
- Provider has correct `host_url` and `api_key_secret_name` configuration
