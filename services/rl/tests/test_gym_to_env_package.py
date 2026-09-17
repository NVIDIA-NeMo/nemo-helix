# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "grpo-examples" / "gym_to_env_package.py"
SPEC = importlib.util.spec_from_file_location("gym_to_env_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _touch_wheels(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).touch()


def test_required_wheel_versions_accept_supported_hydra_stack(tmp_path: Path) -> None:
    _touch_wheels(
        tmp_path,
        "hydra_core-1.3.2-py3-none-any.whl",
        "omegaconf-2.3.0-py3-none-any.whl",
    )

    MODULE.validate_required_wheel_versions(tmp_path)


def test_required_wheel_versions_reject_backtracked_hydra_stack(tmp_path: Path) -> None:
    _touch_wheels(
        tmp_path,
        "hydra_core-0.11.3-py3-none-any.whl",
        "omegaconf-1.4.1-py3-none-any.whl",
    )

    with pytest.raises(SystemExit, match="hydra-core.*0.11.3.*omegaconf.*1.4.1"):
        MODULE.validate_required_wheel_versions(tmp_path)
