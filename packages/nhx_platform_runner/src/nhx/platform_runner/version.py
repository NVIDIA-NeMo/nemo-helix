# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform version and revision helpers."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        return version("nemo-helix").strip()
    except PackageNotFoundError:
        pass
    return os.environ.get("NHX_PLATFORM_VERSION", "dev").strip() or "dev"


__version__ = _resolve_version()


def get_platform_version() -> str:
    return __version__


def get_revision() -> str:
    return os.environ.get("NHX_CODE_REVISION", "dev").strip() or "dev"
