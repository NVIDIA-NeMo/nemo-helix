# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache numpy==2.5.5 opencv-python-headless==5.1.0.95
COPY workload.py captions.py /app/
ENTRYPOINT ["python", "/app/workload.py"]
