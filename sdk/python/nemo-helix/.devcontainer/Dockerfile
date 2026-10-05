# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

USER vscode

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN echo "[[ -d .venv ]] && source .venv/bin/activate || export PATH=\$PATH" >> /home/vscode/.bashrc
