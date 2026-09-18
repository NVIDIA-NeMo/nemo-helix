<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ZITADEL Kubernetes Runtime

This directory contains Kubernetes-specific documentation for the ZITADEL
reference example. The deployment uses the umbrella Helm chart at
`contrib/auth/zitadel/helm`.

The chart is self-seeding for local demos: a post-install job configures
ZITADEL, patches NeMo Platform with generated client IDs, and stores generated
client secrets in the `nemo-zitadel-seed-state` Secret.

For architecture and wiring details, see:

- [Implementation Details](implementation-details.md)
