# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_data_designer_plugin.jobs import bridge, retrieval_bridge
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job_context import JobContext
from pydantic import BaseModel

from tests.task_client_identity import Mode, assert_task_identity, task_environment


@pytest.mark.parametrize("mode", ["headers", "token-file"])
def test_create_entrypoint_uses_creator_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode
) -> None:
    with task_environment(monkeypatch, tmp_path, mode) as capture:
        capture.config_path.write_text(
            json.dumps(
                {
                    "job_config": {"num_records": 1, "config": {"columns": []}},
                    "model_providers": [],
                    "model_configs": [],
                }
            )
        )
        with (
            patch.object(bridge, "_get_ctx", side_effect=capture.context),
            patch.object(bridge, "run_step_config", return_value={"exit_code": 0}),
        ):
            bridge.run()
        assert_task_identity(capture, mode, "data-designer")


class Spec(BaseModel):
    value: str = "test"


class Job:
    def run(self, config: dict[str, str], *, ctx: JobContext, sdk: NemoClient) -> dict[str, int]:
        return {"exit_code": 0}


@pytest.mark.parametrize("mode", ["headers", "token-file"])
def test_retrieval_entrypoint_uses_creator_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode
) -> None:
    with task_environment(monkeypatch, tmp_path, mode) as capture:
        capture.config_path.write_text("{}")
        with patch.object(retrieval_bridge, "_get_ctx", side_effect=capture.context):
            assert retrieval_bridge.run_job_module(Job, Spec) == 0
        assert_task_identity(capture, mode, "data-designer")
