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
Use the image tag matching your NeMo Helix release. See the
[installation guide](https://docs.nvidia.com/nemo-helix/documentation/kubernetes-deployment/setup/helm/install/),
[Gym evaluation guide](https://docs.nvidia.com/nemo-helix/documentation/evaluate-models/agent-eval/gym-runner/), and
[support matrix](https://docs.nvidia.com/nemo-helix/documentation/reference/support-matrix/) for setup,
examples, and resource requirements.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

[NeMo Helix on GitHub](https://github.com/NVIDIA-NeMo/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
It includes open source components under their own licenses, among them
[BusyBox](https://busybox.net), under the GNU General Public License v2. The source code of these
components is in the image under `/usr/share/nemo-helix/source-distributions`.
