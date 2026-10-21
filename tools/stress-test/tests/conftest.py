# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = _ROOT / "worker"
for _path in (_WORKER, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
