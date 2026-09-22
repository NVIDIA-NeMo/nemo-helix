# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-access helpers shared by the CLI commands and credential resolution."""

from __future__ import annotations

import os

from nemo_platform_plugin.client.client import NemoClient


def base_url() -> str:
    """Resolve the platform base URL (matches repo convention NMP_BASE_URL / NEMO_BASE_URL)."""
    return (os.environ.get("NEMO_BASE_URL") or os.environ.get("NMP_BASE_URL") or "http://localhost:8080").rstrip("/")


def make_sdk(base: str) -> NemoClient:
    """Construct a typed platform client against *base*."""
    return NemoClient(base_url=base)
