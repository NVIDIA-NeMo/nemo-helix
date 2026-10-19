<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-customizer

Router service for `/apis/customization`. Training backends (Automodel, RL, Megatron, …) register as **`nemo.customization.contributors`** entry points (discovered via `nemo_helix_plugin.discovery`).

Registers **`nemo.sdk`** → `customization` for `client.customization.*` (composes contributor SDK modules such as `client.customization.automodel.jobs`).

See [docs/CUSTOMIZATION.md](docs/CUSTOMIZATION.md) for contributor authoring.
