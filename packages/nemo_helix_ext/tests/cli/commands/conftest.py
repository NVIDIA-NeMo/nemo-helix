# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import nemo_helix_ext.local.process as _process_mod
import pytest


@pytest.fixture(autouse=True)
def _reset_scope_prefix_cache():
    """Prevent scope prefix leaking between tests."""
    _process_mod._scope_prefix_cache = None
    yield
    _process_mod._scope_prefix_cache = None
