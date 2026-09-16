---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

allowed-tools: Bash(./sdk/stainless.sh:*)
description: Update the Python SDK for NeMo Helix
---
Update the Python SDK for NeMo Helix by following these steps:
- ensure the OpenAPI spec is up-to-date (ask the user and if needed, run the `/script/generate-openapi-spec.sh` script to regenerate it)
- run `./sdk/stainless.sh sync` to update the SDK code from the OpenAPI spec
- troubleshoot any issues using information in @sdk/README.md