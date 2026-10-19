# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from nemo_helix_ext.local.install import services_extra_install_command


@pytest.mark.parametrize(
    ("uv_tool_env", "expected"),
    [
        (False, "pip install 'nemo-helix[all]'"),
        (True, "uv tool install 'nemo-helix[all]'"),
    ],
)
def test_services_extra_install_command_matches_install_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uv_tool_env: bool, expected: str
) -> None:
    if uv_tool_env:
        (tmp_path / "uv-receipt.toml").write_text("[tool]\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert services_extra_install_command() == expected
