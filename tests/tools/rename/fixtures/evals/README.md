<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evals pre-rename snapshots

These inputs were captured from commit `dcc9d2845493c8a0a9ea48b9d7fac83f19593142`
before applying the Evals rename. They preserve the plugin, typed client and
Studio inputs used by the profile integration tests. The `.txt` suffix marks
them as test data; the tests restore their original paths inside disposable
Git repositories.

The Evals profile excludes `tests/tools/rename/**`, so these snapshots remain
unchanged when the product rename is applied. Tests must read these fixed inputs
instead of live product files so they still exercise the pre-rename names after
the real plugin directories and job-source filters have changed.

Update snapshots deliberately when extending the rename profile. Keep their
pre-rename names; do not regenerate them from an already-renamed checkout.
