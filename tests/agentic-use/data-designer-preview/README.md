<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Preview - Harbor Test (Hard)

> **Note:** This eval uses a mix of CLI and Python client. The `nemo data-designer preview`
> CLI command does not exist — the agent must discover this and use the Python client
> (`DataDesignerConfigBuilder` + `client.data_designer.preview()`) to run the preview.
> Model registration also requires the SDK (`inference.providers.update_status`).
> Only secrets and provider creation use the CLI.

Tests an agent's ability to set up inference infrastructure and run a Data Designer
preview with sampler columns, column relationships, and LLM-generated text.

## What This Tests

- Setting up an inference provider (secret, provider, served model registration)
- Constructing a Data Designer configuration with sampler and LLM text columns
- Configuring column relationships (subcategory depends on category)
- Running a preview via the Python client and verifying output

## Columns

| Column | Type | Details |
|--------|------|---------|
| `product_category` | category sampler | Electronics, Clothing, Books, Home |
| `product_subcategory` | subcategory sampler | Depends on product_category |
| `price` | uniform sampler | Float values between 5.0 and 500.0 |
| `status` | category sampler | in_stock, out_of_stock, discontinued |
| `product_description` | LLM text | Generated via real inference |

## Variants

- **`data-designer-preview`** (this, hard) — English-only instructions, agent discovers APIs on its own
- **`data-designer-preview-easy`** — Same task with full CLI/SDK command hints

## Running

```bash
export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
harbor run -p tests/agentic-use/data-designer-preview \
    --agent claude-code \
    --model aws/anthropic/bedrock-claude-sonnet-4-5-v1
```
