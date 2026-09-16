# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The NeMo Fabric Harbor agent installs Fabric and the deepagents harness into the task container
# with `python3 -m venv` + pip, so the image needs CPython, bash, and CA certificates.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
