# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the `evaluation` package importable for its tests.

`evaluation/` is maintainer tooling at the repo root (not shipped under `src/`), so its
tests import `evaluation.*` and need the repo root on `sys.path`. Scoped here — to the
evaluation tests only — rather than widening the global `pythonpath` in pyproject.toml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
