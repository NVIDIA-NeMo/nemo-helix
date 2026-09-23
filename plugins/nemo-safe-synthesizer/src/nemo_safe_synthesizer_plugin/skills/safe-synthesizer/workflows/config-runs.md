<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to run PII-only and generation runs

## Prerequisites

- Read `workflows/config.md` for the job spec field reference.
- Resolve the CLI with `workflows/run.md`.
- Use a platform fileset `data_source`.

## Basic Job Spec

=== "Python SDK"

    ```python
    spec = {
        "data_source": "default/my-input#input.csv",
        "config": {},
    }
    ```

=== "CLI"

    ```json
    {
      "data_source": "default/my-input#input.csv",
      "config": {}
    }
    ```

## PII-only Run

=== "Python SDK"

    ```python
    spec = {
        "data_source": "default/my-input#input.csv",
        "config": {
            "enable_synthesis": False,
            "enable_replace_pii": True,
        },
    }
    ```

=== "CLI"

    ```json
    {
      "data_source": "default/my-input#input.csv",
      "config": {
        "enable_synthesis": false,
        "enable_replace_pii": true
      }
    }
    ```

## Generation without PII Replacement

=== "Python SDK"

    ```python
    spec = {
        "data_source": "default/my-input#input.csv",
        "config": {
            "enable_synthesis": True,
            "enable_replace_pii": False,
            "generation": {"num_records": 100},
            "privacy": {"dp_enabled": False},
            "evaluation": {"enabled": True},
        },
    }
    ```

=== "CLI"

    ```json
    {
      "data_source": "default/my-input#input.csv",
      "config": {
        "enable_synthesis": true,
        "enable_replace_pii": false,
        "generation": {
          "num_records": 100
        },
        "privacy": {
          "dp_enabled": false
        },
        "evaluation": {
          "enabled": true
        }
      }
    }
    ```

## PII Classification Provider

=== "Python SDK"

    ```python
    spec = {
        "data_source": "default/my-input#input.csv",
        "hf_token_secret": "hf-token",
        "config": {
            "replace_pii": {
                "globals": {
                    "classify": {
                        "classify_model_provider": "default/my-nim",
                    },
                },
                "steps": [{}],
            },
        },
    }
    ```

=== "CLI"

    ```json
    {
      "data_source": "default/my-input#input.csv",
      "hf_token_secret": "hf-token",
      "config": {
        "replace_pii": {
          "globals": {
            "classify": {
              "classify_model_provider": "default/my-nim"
            }
          },
          "steps": [{}]
        }
      }
    }
    ```

## Create the Job

Use the CLI, Jobs API, or SDK for the primary platform-container path. Pass the JSON object above as the CLI spec file or as the `spec` field in the create-job request. Confirm the service is configured for container execution before job creation:

```bash
export NHX_IMAGE_REGISTRY=nvcr.io/nvidia/nemo-helix
export NHX_IMAGE_TAG=<tag>  # match your installed NeMo Helix release
export NEMO_SAFE_SYNTHESIZER_CONTAINER_IMAGE=nhx-safe-synthesizer-tasks
```

```bash
nemo safe-synthesizer generate \
  --workspace default \
  --spec-file nss-job.json
```

## Next Steps

- Use `workflows/pii-architecture.md` for provider resolution details.
- Use `workflows/results.md` to retrieve outputs.
- Use `workflows/artifacts.md` to interpret `summary` and `summary.json` first.
