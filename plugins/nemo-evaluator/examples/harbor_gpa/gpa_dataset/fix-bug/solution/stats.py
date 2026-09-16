# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny statistics helpers."""


def mean(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    total = 0
    for value in values:
        total += value
    return total / (len(values) - 1)
