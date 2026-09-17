# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from nmp.automodel.tasks.retrieval_mine.launch import run_hard_negative_mining


def test_run_hard_negative_mining_inherits_stdio(tmp_path: Path) -> None:
    config = tmp_path / "mining_config.yaml"
    config.write_text("mining: {}\n", encoding="utf-8")
    completed = Mock(returncode=0)

    with patch("nmp.automodel.tasks.retrieval_mine.launch.subprocess.run", return_value=completed) as run:
        run_hard_negative_mining(config_file=config, nproc_per_node="1")

    kwargs = run.call_args.kwargs
    assert kwargs.get("capture_output") is not True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    assert kwargs["check"] is True
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    cmd = run.call_args.args[0]
    assert "torch.distributed.run" in cmd
    assert cmd[cmd.index("--nproc_per_node") + 1] == "1"
    assert str(config) in cmd


def test_run_hard_negative_mining_raises_on_nonzero(tmp_path: Path) -> None:
    config = tmp_path / "mining_config.yaml"
    config.write_text("mining: {}\n", encoding="utf-8")

    with (
        patch(
            "nmp.automodel.tasks.retrieval_mine.launch.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["torchrun"]),
        ),
        pytest.raises(RuntimeError, match="Hard-negative mining failed: exit 1"),
    ):
        run_hard_negative_mining(config_file=config, nproc_per_node="1")
