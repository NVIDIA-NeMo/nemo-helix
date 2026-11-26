<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuring Safe Synthesizer

Reference for the job spec fields the plugin accepts. For runnable examples, read `workflows/config-runs.md`. For provider resolution details, read `workflows/pii-architecture.md`.

## Job Spec Schema

```json
{
  "data_source": "default/my-input#input.csv",
  "config": {}
}
```

## Plugin Convenience Flags

The plugin accepts these fields inside `config` and maps them into the Safe Synthesizer runtime config:

- `enable_synthesis`: set to `false` to run only data processing / PII replacement without training and generation.
- `enable_replace_pii`: set to `false` to disable the default PII replacement pipeline.

See `workflows/config-runs.md` for "How to run PII-only and generation runs".

## Data Sources

- Platform jobs: use fileset URLs, e.g. `default/customer-data#input.csv`.
- Supported fileset payload forms include CSV, Parquet, JSON, JSONL, and Hugging Face datasets paths.

## PII Model Provider

PII replacement is on by default and discovers its plan with an LLM. Set the top-level `pii_model_provider` as `<workspace>/<provider_name>` and the model it serves in `config.replace_pii.llm.model_id`. The job fails at creation if either is missing.

```json
{
  "data_source": "default/my-input#input.csv",
  "pii_model_provider": "default/my-nim",
  "config": {
    "replace_pii": {
      "llm": {
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b"
      }
    }
  }
}
```

An inline `config.replace_pii.replacement_plan` makes no LLM request and needs neither field. Set `config.enable_replace_pii` to `false` to turn PII replacement off.

See `workflows/pii-architecture.md` for "PII model provider architecture".

## Secrets

Use `hf_token_secret` at the top level of the job spec when model initialization needs a Hugging Face token from the platform secrets service.

## Reusing a Previously Trained NSS Model

Set `pretrained_model_job` at the top level of the job spec; the plugin resolves that job's `adapter` result from Files for generation-only reuse. Use either `pretrained_model_job` or `config.training.pretrained_model`, not both.
