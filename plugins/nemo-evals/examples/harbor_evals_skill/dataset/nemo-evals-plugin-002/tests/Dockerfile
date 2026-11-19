# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72

RUN mkdir -p /var/lib/helix-task/evidence
WORKDIR /tests
COPY grade.py /tests/
COPY --chmod=0755 test.sh /tests/test.sh
