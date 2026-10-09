---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Sandbox image for NeMo Helix container image builds, part of NeMo Helix
labels: [NeMo]
---
## NeMo Helix Kaniko Container

This container is the sandbox a NeMo Helix image build runs its Dockerfiles in. It is the debug image
of the osscontainertools fork of kaniko, unchanged, with a BusyBox shell.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

### License

This container includes [kaniko](https://github.com/osscontainertools/kaniko) and the registry
credential helpers it ships,
[docker-credential-gcr](https://github.com/googlecloudplatform/docker-credential-gcr),
[amazon-ecr-credential-helper](https://github.com/awslabs/amazon-ecr-credential-helper) and
[docker-credential-acr](https://github.com/osscontainertools/docker-credential-acr), under the Apache
License 2.0; [tini](https://github.com/krallin/tini), under the MIT License; and
[BusyBox](https://busybox.net), under the GNU General Public License v2.
