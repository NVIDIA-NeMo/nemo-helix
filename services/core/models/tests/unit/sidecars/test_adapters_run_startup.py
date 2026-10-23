# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the adapters sidecar starting without NIM_PEFT_SOURCE."""

import logging
import threading
from unittest.mock import patch

import pytest
from nhx.core.models.sidecars.adapters import main as adapters_main


def test_run_without_nim_peft_source_is_a_graceful_noop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """With NIM_PEFT_SOURCE unset, run() must warn and return instead of raising."""
    monkeypatch.delenv("NIM_PEFT_SOURCE", raising=False)
    stop_signal = threading.Event()

    with (
        patch.object(adapters_main, "get_platform_config"),
        patch.object(adapters_main, "Loop") as loop_cls,
        caplog.at_level(logging.WARNING, logger=adapters_main.logger.name),
    ):
        adapters_main.run(parent_stop_signal=stop_signal)

    loop_cls.assert_not_called()
    assert "NIM_PEFT_SOURCE is not set" in caplog.text
