# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.automodel.tasks.training.utils import generate_torchrun_flags_from_env


@pytest.mark.parametrize(
    ("world_size", "expects_no_restarts"),
    [("1", False), ("2", True)],
)
def test_multinode_torchrun_exits_on_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
    world_size: str,
    expects_no_restarts: bool,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("NODE_RANK", "0")

    flags = generate_torchrun_flags_from_env()

    assert ("--max_restarts" in flags) is expects_no_restarts
    if expects_no_restarts:
        assert flags[flags.index("--max_restarts") + 1] == "0"
