# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_agent_optimization_plugin.schemas.optimize import OptimizeSpec, is_fileset_relative
from pydantic import ValidationError


@pytest.mark.parametrize("strategy", ["nat", "prompt-master", "experimentalist"])
def test_optimize_spec_leaves_the_agent_requirement_to_the_strategy(strategy: str) -> None:
    """The generic spec names no strategy: whether ``agent`` is required is strategy-defined.

    Each strategy enforces its own rule in ``validate_config`` (see
    ``test_strategy_requires_a_platform_agent`` in prompt-master and
    ``test_validate_config_rejects_missing_agent`` in switchyard), so a third-party strategy
    that needs no agent does not have to edit this schema.
    """
    spec = OptimizeSpec(strategy=strategy, optimize_config="/tmp/config.yaml", workspace="default")
    assert spec.agent is None


def test_optimize_spec_rejects_an_empty_strategy() -> None:
    with pytest.raises(ValidationError, match="strategy"):
        OptimizeSpec(strategy="", optimize_config="/tmp/config.yaml", workspace="default")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("configs/x.yaml", True), ("/abs/path.yaml", False), ("../escape.yaml", False)],
)
def test_is_fileset_relative(value: str, expected: bool) -> None:
    assert is_fileset_relative(value) is expected
