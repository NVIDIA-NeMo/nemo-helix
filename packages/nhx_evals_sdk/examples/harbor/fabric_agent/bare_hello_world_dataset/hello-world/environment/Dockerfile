# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# A deliberately bare task image: no Python, no pip, no curl, and no NeMo Fabric. Everything the
# harness needs is provisioned by `FabricInstalledAgent`, which is the point of this example. Only
# bash (for Harbor's exec) comes from the image, and Ubuntu already has it.
FROM ubuntu:24.04

WORKDIR /app
