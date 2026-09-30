# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal terminal formatting used by the workflow.

``workflow.py`` uses ``Console`` for numbered steps, aligned details, success
messages, and warnings. Centralizing output here prevents implementation
modules from inventing inconsistent machine-like status strings.
"""

from __future__ import annotations

import sys
from typing import TextIO


class Console:
    """Render concise, human-readable workflow progress."""

    def __init__(self, *, output: TextIO = sys.stdout, error_output: TextIO = sys.stderr) -> None:
        """Create a console that writes normal and warning messages separately."""
        self.output = output
        self.error_output = error_output

    def step(self, number: int, total: int, message: str) -> None:
        """Print the heading for a workflow step."""
        print(f"\n[{number}/{total}] {message}", file=self.output, flush=True)

    def detail(self, label: str, value: object) -> None:
        """Print an indented label and value below the current step."""
        print(f"      {label}: {value}", file=self.output, flush=True)

    def success(self, message: str) -> None:
        """Print the final success message."""
        print(f"\nSUCCESS: {message}", file=self.output, flush=True)

    def warning(self, message: str) -> None:
        """Print a warning without hiding the workflow's primary failure."""
        print(f"\nWARNING: {message}", file=self.error_output, flush=True)
