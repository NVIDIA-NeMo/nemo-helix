# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = _ROOT / "worker"
for _path in (_ROOT, _WORKER):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
sys.path.insert(0, str(_WORKER))
sys.path.insert(0, str(_ROOT))
