<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# jobs-multistep-cpu-cli

## Overview

This eval tests **real job execution** through the NeMo Helix jobs pipeline. The agent creates three jobs (success, intentional failure, recovery), polls each to terminal status, diagnoses the failure, and demonstrates understanding of the jobs lifecycle.

## What it tests

- Full job lifecycle: create -> execute -> complete/error
- Error handling: diagnosing a failed job (exit code 1)
- Recovery: successfully creating and running a job after a failure
- Multiple image support (busybox and alpine)

## How Docker socket access works

The Harbor framework supports per-task `docker-compose.yaml` overrides. The `environment/docker-compose.yaml` in this eval adds a bind mount for the Docker socket:

```yaml
services:
  main:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

This is merged with Harbor's base compose files, giving the NeMo Helix API server access to the host Docker daemon via the DOOD (Docker-outside-of-Docker) pattern.

**Prerequisite**: The host machine must have a Docker socket at `/var/run/docker.sock`.
