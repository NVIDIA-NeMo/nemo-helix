# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_agent_optimization_plugin.schemas.optimize import OptimizeSpec, is_fileset_relative
from pydantic import ValidationError


def test_optimize_spec_requires_agent_for_non_nat_strategy() -> None:
    with pytest.raises(ValidationError, match="agent is required"):
        OptimizeSpec(strategy="prompt-master", optimize_config="/tmp/config.yaml", workspace="default")


def test_optimize_spec_allows_missing_agent_for_nat() -> None:
    spec = OptimizeSpec(strategy="nat", optimize_config="/tmp/config.yaml", workspace="default")
    assert spec.agent is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("configs/x.yaml", True), ("/abs/path.yaml", False), ("../escape.yaml", False)],
)
def test_is_fileset_relative(value: str, expected: bool) -> None:
    assert is_fileset_relative(value) is expected
