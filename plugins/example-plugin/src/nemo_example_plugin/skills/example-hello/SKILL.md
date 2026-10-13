---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: example-hello
description: Demonstrates a minimal plugin-provided skill from the example plugin.
version: 0.1
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Example Hello

Use this dummy skill to verify that `nemo skills list`, `show`, and `install`
can load plugin-provided skills through the `nemo.skills` entry-point surface.
