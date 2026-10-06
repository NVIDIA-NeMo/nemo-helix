# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared printing utilities for NeMo Helix SDK tools.

This module provides common printing functions with color support
that can be used across the SDK tools modules.
"""

from rich import get_console
from rich.padding import Padding
from rich.text import Text


def print_color(s: str, color: str = "green", padding: tuple[int, int] | None = None):
    """
    Print text with color styling using rich.

    Args:
        s: The string to print
        color: The color to use (e.g., 'green', 'red', 'yellow', 'cyan')
        padding: Optional padding tuple (top/bottom, left/right) to add around the text

    Examples:
        >>> print_color("Success!", "green")
        >>> print_color("Warning", "yellow", padding=(1, 2))
        >>> print_color("Error occurred", "red")
    """
    t = Text(s)
    t.stylize(style=color)
    if padding:
        t = Padding(t, padding)
    get_console().print(t)
