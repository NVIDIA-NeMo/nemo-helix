---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Sandboxed host runtime for NeMo Gym evaluations, part of NeMo Helix
labels: [NeMo]
---
## NeMo Helix Gym Host Container

This container runs NeMo Gym environments inside a sandbox provisioned by the
NeMo Helix Evals. It includes NeMo Gym and the sandbox host runtime, and
prefetches first-party Gym component environments when available.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

[NeMo Helix on GitHub](https://github.com/NVIDIA-NeMo/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
It includes open source components under their own licenses. Their source code is in the image under
`/usr/share/nemo-helix/source-distributions`.
