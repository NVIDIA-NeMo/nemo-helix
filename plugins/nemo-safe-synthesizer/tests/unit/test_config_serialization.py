# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep omitted config defaults out of persisted and dispatched job specs."""

import json

import pytest
from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig
from nemo_safe_synthesizer_plugin.jobs.generate import _task_job_config


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"evaluation": {}},
        {"evaluation": {"mia_excluded_columns": None}},
        {"evaluation": {"mia_excluded_columns": ["sensitive"]}},
        {"replace_pii": None, "privacy": None},
        {"unknown_fields": "reject", "emit_telemetry": True},
        {"generation": {"num_records": 25}},
    ],
)
def test_config_omissions_survive_serialization_and_reload(config):
    spec = SafeSynthesizerJobConfig(data_source="default/data#input.csv", config=config)

    # Storage must not expand omitted fields into explicit defaults.
    stored_json = spec.model_dump_json()
    assert json.loads(stored_json)["config"] == config

    # Reload the stored spec, then serialize it for task dispatch.
    reloaded_spec = SafeSynthesizerJobConfig.model_validate_json(stored_json)
    dispatched_spec = _task_job_config(reloaded_spec)
    assert dispatched_spec["config"] == config
