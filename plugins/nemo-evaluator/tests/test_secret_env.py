# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task's secret-sourced environment."""

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.secret_env import build_task_environment


def test_two_secrets_under_one_env_name_are_refused_naming_both() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_task_environment([("OPENAI_API_KEY", "team-a/openai-key"), ("OPENAI_API_KEY", "team-b/openai-key")])
    assert str(excinfo.value) == (
        "conflicting secret references for environment variable 'OPENAI_API_KEY': "
        "'team-a/openai-key' and 'team-b/openai-key'"
    )


def test_one_secret_under_several_consumers_is_emitted_once() -> None:
    environment = build_task_environment([("OPENAI_API_KEY", "openai-key"), ("OPENAI_API_KEY", "openai-key")])
    assert [variable.name for variable in environment].count("OPENAI_API_KEY") == 1
