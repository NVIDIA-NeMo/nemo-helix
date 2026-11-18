# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors the node adapters share with the platform client."""

from __future__ import annotations


class Missing(LookupError):
    """The external resource does not exist yet. The tick asks ``start`` to create it."""
