---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Agent installer for NeMo Helix evaluation agent layers, part of NeMo Helix
labels: [NeMo]
---
## NeMo Helix Agent Installer Container

This container holds the installer a NeMo Helix evaluation run uses to build an agent layer: a task's
environment image with the run's agent already installed. It is an archive of a standalone CPython and a
relocatable virtual environment with Harbor, NeMo Fabric's Harbor agent and the NeMo Helix agent installer,
which each layer build mounts and runs inside the task's image.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

[NeMo Helix on GitHub](https://github.com/NVIDIA-NeMo/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
The source code of the third-party components it includes is in the image under
`/usr/share/nemo-helix/source-distributions/nhx-agent-installer`.
