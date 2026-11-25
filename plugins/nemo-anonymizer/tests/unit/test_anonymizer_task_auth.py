# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_anonymizer_plugin.tasks.anonymizer import run as task
from nemo_helix_plugin.client.client import NemoClient

from tests.task_client_identity import Mode, assert_task_identity, task_environment


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "request": {
                    "config": {"replace": {"kind": "redact"}},
                    "data": {"source": "https://example.test/input.csv", "text_column": "text"},
                },
                "model_configs_yaml": "model_configs: []",
                "dd_model_providers": [],
            }
        )
    )


@pytest.mark.parametrize("mode", ["headers", "token-file"])
def test_entrypoint_uses_creator_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode) -> None:
    with task_environment(monkeypatch, tmp_path, mode) as capture:
        _write_config(capture.config_path)
        with (
            patch.object(task, "_get_ctx", side_effect=capture.context),
            patch.object(task, "run_step_config", return_value=0),
        ):
            assert task.run() == 0
        assert_task_identity(capture, mode, "anonymizer")


def test_entrypoint_keeps_injected_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with task_environment(monkeypatch, tmp_path, "headers") as capture:
        _write_config(capture.config_path)
        with (
            NemoClient(base_url="https://localhost", default_headers={"custom": "unchanged"}) as injected,
            patch.object(task, "_get_ctx", side_effect=capture.context),
            patch.object(task, "run_step_config", return_value=0),
        ):
            assert task.run(sdk=injected) == 0
            assert capture.clients == [injected]
            assert injected.default_headers == {"custom": "unchanged"}
