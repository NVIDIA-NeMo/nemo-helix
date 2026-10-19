# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Matches the tag CI pre-pulls in the integration job, so building this task image reuses that
# layer
FROM alpine:3.23

# bash is required for Harbor's docker environment
RUN apk add --no-cache bash

WORKDIR /app
