<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vendored from NVIDIA-NeMo/labs-eval-author

- Repository: https://github.com/NVIDIA-NeMo/labs-eval-author
- Commit: `a0f55d166eccda6e1417f2160276e98f43feeb52`
- Contents: `LICENSE`, `NOTICE`, and these `skills/` directories, each byte-identical to upstream:
  `ethos`, `eval-author`, `eval-author-audit`, `eval-author-discover`, `eval-author-environment`,
  `eval-author-first-eval`, `eval-author-task-create`, `eval-author-trace-environment`, `gym-to-atif`.
- Omitted: `eval-author-adapt`, `eval-author-inspect-trace`, `mlflow-to-atif`. No file reachable by a
  link from `eval-author-first-eval/SKILL.md` references them.

Do not edit the vendored files. To update, re-vendor from upstream and re-apply the omission. Only
`eval-author-first-eval` is exposed by this plugin; the other directories ship because the first-eval
skill links into them, and `tests/test_vendor.py` checks that every linked file is present.
