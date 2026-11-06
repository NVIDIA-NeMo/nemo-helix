---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: API component of NeMo Helix
---
## NeMo Helix API Container

This container image provides the NeMo Helix REST API for inference, models, files, jobs, evaluations, guardrails, and observability. Use it to develop and operate AI agents through the NeMo Helix CLI, Python SDK, or NeMo Studio. It is a CPU container that supports local Docker use and self-managed Kubernetes deployments; see the [NeMo Helix support matrix](https://docs.nvidia.com/nemo-helix/documentation/reference/support-matrix) for supported hosts and resource requirements.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

[NeMo Helix on GitHub](https://github.com/NVIDIA-NeMo/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
It includes open source components under their own licenses, among them
[BusyBox](https://busybox.net), under the GNU General Public License v2. The source code of these
components is in the image under `/usr/share/nemo-helix/source-distributions`.
