# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared exceptions for plugin-local execution paths."""


class LocalRunError(RuntimeError):
    """Raised for local invocation setup errors that should be shown as CLI input errors."""


__all__ = ["LocalRunError"]
