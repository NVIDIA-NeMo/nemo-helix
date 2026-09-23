<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Preview

You have access to the `nhx` CLI and the NeMo Helix Python SDK for NeMo Helix operations. Your task is to set up inference and generate a preview of synthetic data using the Data Designer with both sampler and LLM-generated columns.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The Python SDK is available at `/app/.venv/bin/python` with `from nemo_helix import NeMoHelix`. Both connect to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

The `ANTHROPIC_API_KEY` environment variable contains an API key that works with NVIDIA's inference API at `https://inference-api.nvidia.com/v1`.

## Task

### Step 1: Set up inference provider

Create a secret named `nvidia-api-key` using the value of `$ANTHROPIC_API_KEY`, then create an inference provider named `nvidia-inference` pointing to `https://inference-api.nvidia.com/v1` with that secret. Finally, register a served model so the model entity `default/text-llm` maps to the upstream model `aws/anthropic/bedrock-claude-sonnet-4-5-v1`.

### Step 2: Build Data Designer configuration

Create a Data Designer configuration with these columns:
- A **category** sampler column named `product_category` that samples from: `"Electronics"`, `"Clothing"`, `"Books"`, `"Home"`
- A **subcategory** sampler column named `product_subcategory` that depends on `product_category`, with these mappings:
  - `"Electronics"` → `["Smartphones", "Laptops", "Headphones"]`
  - `"Clothing"` → `["Shirts", "Pants", "Shoes"]`
  - `"Books"` → `["Fiction", "Non-Fiction", "Technical"]`
  - `"Home"` → `["Furniture", "Kitchen", "Decor"]`
- A **uniform** distribution sampler column named `price` that samples floating-point values between 5.0 and 500.0
- A **category** sampler column named `status` that samples from: `"in_stock"`, `"out_of_stock"`, `"discontinued"`
- An **LLM text** column named `product_description` that uses the model alias `text` to generate a one-sentence product description. The prompt should reference the `product_category` and `product_subcategory` columns using Jinja2 template syntax (e.g., `{{ product_category }}`).

Include a model configuration that maps alias `text` to model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` with provider `default/nvidia-inference`. Set `skip_health_check: true` on the model config.

### Step 3: Run preview and verify output

Run a Data Designer preview using this configuration, requesting 10 records. Verify that the preview produces output containing all expected columns, including LLM-generated product descriptions.

## Notes

- The workspace to use is `default`.
- You may use either the CLI or the Python SDK to run the preview — use whichever is available and works.

## Success Criteria

The task is complete when:
- The inference provider `nvidia-inference` is set up with the served model
- A Data Designer preview has been successfully executed with all five columns
- The preview generated records containing `product_category`, `product_subcategory`, `price`, `status`, and `product_description` columns
- The `product_subcategory` values are consistent with their parent `product_category`
- The `product_description` column contains non-empty LLM-generated text
