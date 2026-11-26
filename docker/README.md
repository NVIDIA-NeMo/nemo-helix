<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Docker Builds

Dockerfiles and Docker build support files live under this directory. The
canonical Buildx Bake file is `docker-bake.hcl`.

Use `docker-api`, `docker-tasks`, `docker-build-tools`, or `docker-gym-host`
to build one CPU image family. The corresponding `-ci` groups add the smoke
stages or the DeepAgents E2E image. `docker-cpu` builds the API, tasks, and
tooling images together. `docker-cpu-ci` also includes Gym host and CI targets.

Examples:

```bash
docker buildx bake --print docker-cpu
docker buildx bake --print nhx-automodel
docker buildx bake --print nhx-unsloth
```
