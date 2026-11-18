# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from nhx.automodel.tasks.retrieval_mine import __main__ as task

from tests.task_client_identity import Mode, assert_task_identity, task_environment


@pytest.mark.parametrize("mode", ["headers", "token-file"])
def test_mining_entrypoint_uses_creator_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode
) -> None:
    with task_environment(monkeypatch, tmp_path, mode) as capture:
        capture.config_path.write_text(json.dumps({"job_config": {}, "model_fileset": "allowed/model"}))
        with (
            patch.object(task, "_get_ctx", side_effect=capture.context),
            patch.object(task, "run_mine", return_value={"exit_code": 0}),
        ):
            assert task.main() == 0
        assert_task_identity(capture, mode, "data-designer")
