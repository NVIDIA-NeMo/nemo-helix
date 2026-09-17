# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make Automodel's tqdm bars readable in job logs.

Shared by the training and hard-negative-mining entry points: both run recipes
that draw tqdm bars, and both have their stdout scraped per line by the jobs log
sidecar. Lives above the task packages because ``tqdm`` is only present in the
training images, so this cannot move to ``nmp_customization_common``.
"""

from __future__ import annotations

import sys

import tqdm as tqdm_module
from tqdm import tqdm as TqdmBar

__all__ = ["LineTqdm", "install_line_tqdm"]


class LineTqdm(TqdmBar):
    """tqdm that writes a full line per refresh.

    Job logs are collected per line, so carriage-return bars arrive as one
    unreadable record.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 15.0)
        kwargs.setdefault("file", sys.stderr)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("disable", False)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        self.fp.write(f"{self}\n")
        self.fp.flush()
        return True


def install_line_tqdm() -> None:
    """Replace ``tqdm.tqdm`` before the recipe builds its bar.

    Automodel resolves ``tqdm`` lazily inside the function that creates the bar,
    so patching the module attribute works even after the recipes are imported.
    """
    tqdm_module.tqdm = LineTqdm  # ty: ignore[invalid-assignment]
