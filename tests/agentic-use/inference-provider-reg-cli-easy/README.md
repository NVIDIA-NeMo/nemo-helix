<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Provider Registration - CLI Eval

## Overview

This eval tests whether a coding agent can register a model provider in NeMo Helix's Inference Gateway (IGW) using the CLI. It covers the basic flow of creating an API key secret and registering a provider that references it.

## What it tests

1. **Secret creation** - Creating an API key secret for provider authentication
2. **Provider registration** - Registering a model provider with host URL and secret reference
3. **Provider listing** - Verifying the provider appears in the list
4. **Provider retrieval** - Getting provider details by name
5. **Provider deletion** - Removing a provider
6. **Final state verification** - A final provider exists with correct configuration

## What it does NOT test

This eval does **not** verify that inference works through the registered provider. The original
flow spec includes "Provider can be used for inference" as a success criterion, but testing actual
inference would require a real API key for an external provider (e.g., build.nvidia.com), which is
not appropriate for an automated eval. Inference verification is better covered by separate evals
(e.g., Flow #9 or #11 with MockLLM).

## Flow Reference

Based on Flow #7 - Model Provider Registration from `tests/agentic-use/agentic_flows/inference.md`.

## Success Criteria

- Secret `harbor-provider-api-key` exists
- Provider `harbor-test-provider` was created and then deleted
- Provider `harbor-final-provider` exists with correct host URL, description, and secret reference
