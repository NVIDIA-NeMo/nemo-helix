<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

/app/stats.py defines `mean(values)`, which is supposed to return the arithmetic mean of a
non-empty list of numbers, but `python3 /app/check.py` currently fails. Fix the bug in
/app/stats.py so that `python3 /app/check.py` exits 0. Do not modify /app/check.py.
