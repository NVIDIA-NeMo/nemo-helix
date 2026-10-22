<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-automodel

Compiler and task entrypoints for NeMo Automodel training jobs on the platform. **No HTTP server** — consumed by `nemo-automodel-plugin` and Jobs task images (`my-registry/nemo-helix-dev/nhx-customizer-tasks`, `.../nhx-automodel-training`).

Training failures are classified into user-facing error types by the rules shared with the other customization backends, in `packages/nhx_customization_common/src/nhx/customization_common/training/errors/` (see its README).
