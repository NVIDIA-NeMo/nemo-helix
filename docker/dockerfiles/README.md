<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Service Dockerfile Parking Area

This directory contains service Dockerfiles that have not been consolidated into
the top-level Dockerfile layout. Only files referenced by `docker-bake.hcl` are
active build inputs.

Currently active:

- `services/guardrails/callouts/Dockerfile.bake` via the `nhx-guardrails-callout-*`
  bake targets.

The other service Dockerfiles are retained for owner review and should be
removed if no owner confirms they are still needed.
