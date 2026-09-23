# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_helix_ext.cli.app import app
from typer.testing import CliRunner

from ..utils import assert_exit_code


def test_list_projects(runner: CliRunner):
    workspace = "default"

    result = runner.invoke(app, f"projects list --workspace {workspace}")
    assert_exit_code(result, 0)

    output = result.stdout
    parsed_output = json.loads(output)

    assert "data" in parsed_output
    assert len(parsed_output["data"]) == 1
    assert parsed_output["data"][0]["id"].startswith("project-")
    assert parsed_output["data"][0]["workspace"] == "default"
