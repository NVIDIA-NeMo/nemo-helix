# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pre_fix_tool() -> dict[str, Any]:
    return json.loads((FIXTURES / "pre-3857.json").read_text())


@pytest.fixture
def post_fix_tool() -> dict[str, Any]:
    return json.loads((FIXTURES / "post-3857.json").read_text())
