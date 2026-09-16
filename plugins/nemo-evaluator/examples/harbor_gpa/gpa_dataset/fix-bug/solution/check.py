# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit 0 iff stats.mean is correct."""

import sys

from stats import mean

cases = [([2, 4, 6], 4.0), ([1], 1.0), ([1.5, 2.5], 2.0), ([10, 20, 30, 40], 25.0)]
failures = [(values, expected, mean(values)) for values, expected in cases if abs(mean(values) - expected) > 1e-9]
for values, expected, actual in failures:
    print(f"mean({values}) = {actual}, expected {expected}")
sys.exit(1 if failures else 0)
