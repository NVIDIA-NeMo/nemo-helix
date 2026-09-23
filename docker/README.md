<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Docker Builds

Dockerfiles and Docker build support files live under this directory. The
canonical Buildx Bake file is `docker-bake.hcl`.

Examples:

```bash
docker buildx bake --print docker-cpu
docker buildx bake --print nhx-automodel
docker buildx bake --print nhx-unsloth
```
