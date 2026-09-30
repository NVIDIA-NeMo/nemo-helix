---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Job runner for Gym agent evaluations, part of NeMo Helix
---
## NeMo Helix Gym Tasks Container

This container provides the Gym and Ray runtime for trusted deployments that run
Gym directly inside an agent-evaluation task. Sandboxed Gym evaluations use the
shared CPU tasks container to orchestrate a separate Gym host, keeping Gym and
Ray outside the task container.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
