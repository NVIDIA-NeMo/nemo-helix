<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII model provider architecture

## Prerequisites

- The job config includes `config.replace_pii`.
- The referenced provider exists in the target workspace and is written as `<workspace>/<provider_name>`.
- Platform jobs can reach the Models service and Inference Gateway.

## Provider Resolution

The plugin resolves the top-level `pii_model_provider` through Inference Gateway during job compilation. It stores only the provider route path in the task environment and lets the task combine that path with the platform Models service URL at runtime.

## Next Steps

- Configure provider fields with `workflows/config.md`.
- Run PII-only examples from `workflows/config-runs.md`.
- Diagnose endpoint failures with `workflows/diagnose.md`.
