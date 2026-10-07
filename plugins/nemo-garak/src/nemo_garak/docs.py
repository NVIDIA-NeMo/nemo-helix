# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docs surface for the garak plugin scaffold."""

from __future__ import annotations

from pathlib import Path


def get_docs_path() -> Path:
    """Return the directory containing plugin docs."""

    return Path(__file__).parent / "docs"
