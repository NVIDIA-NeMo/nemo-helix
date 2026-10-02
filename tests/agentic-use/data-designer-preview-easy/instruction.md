<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Preview

You have access to the `nemo` CLI and the NeMo Helix Python SDK for NeMo Helix operations. Your task is to set up inference and generate a preview of synthetic data using the Data Designer with both sampler and LLM-generated columns.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The Python SDK is available at `/app/.venv/bin/python` with `from nemo_helix import NeMoHelix`. Both connect to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

The `ANTHROPIC_API_KEY` environment variable contains an API key that works with NVIDIA's inference API at `https://inference-api.nvidia.com/v1`.

## Available CLI Commands

### Secrets Commands

- `nemo secrets create <name> --data "<value>" --description "<description>"` - Create a secret directly
- `nemo secrets list` - List all secrets

### Inference Provider Commands

- `nemo inference providers create <name> --host-url <url> --api-key-secret-name <secret_name> --description "<description>"` - Register a model provider
- `nemo inference providers list` - List all registered providers
- `nemo inference providers retrieve <name>` - Get provider details

### Model Registration (Python SDK)

Register a served model so the inference gateway maps a model entity to an upstream model:

```python
from nemo_helix import NeMoHelix

sdk = NeMoHelix(base_url="http://localhost:8080")
sdk.inference.providers.update_status(
    name="<provider_name>",
    workspace="default",
    served_models=[{
        "model_entity_id": "default/<model_entity_name>",
        "served_model_name": "<upstream_model_name>",
    }],
)
```

### Data Designer Preview (Python SDK)

The `nemo data-designer preview` CLI command is not available. Use the Python SDK instead:

```python
import data_designer.config as dd
from nemo_helix import NeMoHelix

client = NeMoHelix(base_url="http://localhost:8080", workspace="default")

# Build config with the ConfigBuilder
config_builder = dd.DataDesignerConfigBuilder(
    model_configs=[
        dd.ModelConfig(
            alias="<alias>",
            model="<upstream_model>",
            provider="default/<provider_name>",
            skip_health_check=True,
        ),
    ]
)

# Add sampler columns
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="<column_name>",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["val1", "val2"]),
    )
)

# Add subcategory column (depends on a parent category column)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="<subcategory_column>",
        sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(
            category="<parent_column_name>",
            values={"parent_val1": ["sub1", "sub2"], "parent_val2": ["sub3", "sub4"]},
        ),
    )
)

# Add uniform distribution column
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="<column_name>",
        sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=0.0, high=100.0),
    )
)

# Add LLM text column (references other columns via Jinja2 templates)
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="<column_name>",
        model_alias="<alias>",
        prompt="Generate text about {{ other_column }}",
    )
)

# Run preview
preview_results = client.data_designer.preview(config_builder, num_records=10)
print(preview_results.dataset)
```

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

## Success Criteria

The task is complete when:
- The inference provider `nvidia-inference` is set up with the served model
- A Data Designer preview has been successfully executed with all five columns
- The preview generated records containing `product_category`, `product_subcategory`, `price`, `status`, and `product_description` columns
- The `product_subcategory` values are consistent with their parent `product_category`
- The `product_description` column contains non-empty LLM-generated text
