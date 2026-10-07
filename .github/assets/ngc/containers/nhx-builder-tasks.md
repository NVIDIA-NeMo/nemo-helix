---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Build steps for NeMo Helix container image builds, part of NeMo Helix
labels: [NeMo]
---
## NeMo Helix Builder Tasks Container

This container runs the steps of a NeMo Helix image build: it fetches the build context, runs the
build in a sandbox, then pushes and signs the images. It includes crane, which pushes them.

### Resources

[Documentation](https://docs.nvidia.com/nemo-helix)

### License

This container is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/LICENSE).
It includes [crane](https://github.com/google/go-containerregistry), under the Apache License 2.0.
