# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core business logic for the evals plugin.

This module contains the actual functionality — independently of how it is
exposed.  The service and CLI are thin wrappers that delegate here.
"""


def say_hello(name: str) -> str:
    """Return a greeting for the given name.

    Args:
        name: The name to greet.

    Returns:
        A friendly greeting string.
    """
    return f"Hello from evals plugin, {name}!"
