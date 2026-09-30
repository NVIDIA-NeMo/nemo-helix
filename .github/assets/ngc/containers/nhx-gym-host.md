---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Sandboxed host runtime for NeMo Gym evaluations, part of NeMo Helix
labels: [NeMo]
---
## NeMo Helix Gym Host Container

This container runs NeMo Gym environments inside a sandbox provisioned by the
NeMo Evaluator. It includes NeMo Gym and the sandbox host runtime, and
prefetches first-party Gym component environments when available.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
