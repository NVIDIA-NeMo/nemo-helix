# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The whole task environment: a bare Python image. The agent is stdlib-only, so
# there is nothing to install and the image builds in seconds.

FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /app/artifacts /app/traces
