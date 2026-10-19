# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Intentional multi-step failure fixture: WrappedAgent uploads the source
# successfully, but `python main.py` fails with exit code 127 because this image
# has no Python. Harbor must capture the wrapper's RuntimeError at
# result.json step_results[0].exception_info while leaving the top-level
# exception_info null.

FROM debian:bookworm-slim

WORKDIR /app

RUN mkdir -p /app/artifacts /app/traces
