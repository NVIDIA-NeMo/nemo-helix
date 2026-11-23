<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Safe Synthesizer Results

## Prerequisites

- Know the platform job name and workspace.

## Platform Jobs

Platform jobs publish named results through the Jobs service:

- `summary`
- `synthetic-data`
- `evaluation-report`
- `evaluation-report-data`
- `adapter`

Use the Jobs API or SDK to list and fetch result records for platform jobs. The plugin does not expose plugin-specific CLI result commands.

## Next Steps

- Interpret artifact names and missing output cases with `workflows/artifacts.md`.
- Check platform job status with the Jobs API or SDK.
- Diagnose failures with `workflows/diagnose.md`.
