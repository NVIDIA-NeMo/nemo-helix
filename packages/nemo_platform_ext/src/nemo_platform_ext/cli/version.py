# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version reporting for the ``nemo`` CLI."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Distributions that can carry the CLI, most specific first. The wrapper
# distribution and the SDK distribution share one release version.
_DISTRIBUTIONS = ("nemo-platform", "nemo-platform-sdk", "nemo-platform-ext")

UNKNOWN_VERSION = "unknown"


def client_version() -> str:
    """Return the installed release version of the CLI, or ``"unknown"``."""
    for distribution in _DISTRIBUTIONS:
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return UNKNOWN_VERSION
