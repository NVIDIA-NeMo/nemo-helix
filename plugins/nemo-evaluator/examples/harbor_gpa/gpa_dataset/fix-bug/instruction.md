<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`stats.py` in your working directory defines `mean(values)`, which is supposed to return the
arithmetic mean of a non-empty list of numbers, but it has a bug: `check.py` next to it fails when
run. Fix the bug in `stats.py`. Do not modify `check.py`.
