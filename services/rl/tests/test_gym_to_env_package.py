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


def test_rl_dependency_policy_reads_uv_tables(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv]
override-dependencies = ["fastapi[standard]>=0.133.0,<0.137.0"]
constraint-dependencies = ["urllib3>=2.7.0"]
""",
        encoding="utf-8",
    )

    overrides, constraints = MODULE.rl_dependency_policy(tmp_path)

    assert overrides == ["fastapi[standard]>=0.133.0,<0.137.0"]
    assert constraints == ["urllib3>=2.7.0"]


def test_rl_dependency_policy_rejects_a_checkout_without_uv_limits(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.uv]\nmanaged = true\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="override-dependencies"):
        MODULE.rl_dependency_policy(tmp_path)


def test_package_info_version_keeps_the_prerelease_suffix(tmp_path: Path) -> None:
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym" / "package_info.py").write_text(
        'MAJOR = 0\nMINOR = 7\nPATCH = 0\nPRE_RELEASE = "rc0"\n',
        encoding="utf-8",
    )

    assert MODULE.nemo_gym_version(tmp_path) == "0.7.0rc0"


def test_locked_versions_use_nemo_gym_openai_when_the_lock_has_two(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(
        """
[[package]]
name = "nemo-gym"
dependencies = [
    { name = "openai", version = "2.44.0" },
    { name = "ray", extra = ["default"] },
]

[[package]]
name = "openai"
version = "2.6.1"

[[package]]
name = "openai"
version = "2.44.0"

[[package]]
name = "ray"
version = "2.56.1"
""",
        encoding="utf-8",
    )

    ray, openai = MODULE.locked_distribution_versions(tmp_path)

    assert ray == "2.56.1"
    assert openai == "2.44.0"
