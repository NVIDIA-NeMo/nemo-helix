# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("harbor")

from nemo_evaluator.examples.harbor_test_agent import AGENT_DIR, WrappedAgent


def test_packaged_wrapper_resolves_example_assets() -> None:
    assert WrappedAgent.name() == "hello-harbor-agent"
    assert AGENT_DIR.name == "agent"
    for name in ("agent.py", "main.py", "tracing.py"):
        assert (AGENT_DIR / name).is_file()
