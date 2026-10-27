# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep config omissions across API storage, reload, PATCH and dispatch."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec, job_route_factory
from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig
from nemo_safe_synthesizer_plugin.jobs.generate import _create_job_step, _task_job_config
from nemo_safe_synthesizer_plugin.sdk.job_builder import _to_dict
from pydantic import ConfigDict, Field, ValidationError, create_model

DATA_SOURCE = "default/data#input.csv"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"evaluation": {}},
        {"data": {}},
        {"training": {}},
        {"data": {"max_sequences_per_example": 10}},
        {"evaluation": {"mia_excluded_columns": None}},
        {"evaluation": {"mia_excluded_columns": ["sensitive"]}},
        {"replace_pii": None, "privacy": None},
        {"unknown_fields": "reject", "emit_telemetry": True},
        {"generation": {"num_records": 25}},
        {"privacy": {"dp_enabled": True}},
        {"emit_telemetry": False},
    ],
)
def test_config_round_trip_preserves_only_supplied_fields(config):
    spec = SafeSynthesizerJobConfig(data_source=DATA_SOURCE, config=config)
    for _ in range(3):
        dumped = spec.model_dump(mode="json")
        assert dumped["config"] == config
        assert json.loads(spec.model_dump_json())["config"] == config
        assert _task_job_config(spec)["config"] == config
        spec = SafeSynthesizerJobConfig.model_validate(json.loads(json.dumps(dumped)))


def test_builder_model_input_keeps_nested_omissions_and_explicit_nulls():
    params = SafeSynthesizerJobConfig(
        data_source=DATA_SOURCE, config={"evaluation": {"mia_excluded_columns": None}}
    ).config
    assert _to_dict(params.evaluation) == {"mia_excluded_columns": None}
    assert _to_dict(params.training) == {}


def test_resolved_library_defaults_do_not_become_explicit_options():
    spec = SafeSynthesizerJobConfig(data_source=DATA_SOURCE, config={"data": {}})
    assert spec.config.data.max_sequences_per_example == 10
    assert spec.config.data.model_fields_set == set()
    assert spec.model_dump()["config"] == {"data": {}}


def test_unknown_config_is_not_silently_dropped():
    with pytest.raises(ValidationError, match="not_a_supported_option"):
        SafeSynthesizerJobConfig(data_source=DATA_SOURCE, config={"evaluation": {"not_a_supported_option": True}})


def test_strict_task_schema_without_new_optional_field():
    # Model a strict consumer without one optional evaluation field. This is
    # not a claim that all configurations work with every older task library.
    spec = SafeSynthesizerJobConfig(data_source=DATA_SOURCE, config={})
    evaluation_type = type(spec.config.evaluation)
    fields: dict[str, Any] = {
        name: (field.annotation, field)
        for name, field in evaluation_type.model_fields.items()
        if name != "mia_excluded_columns"
    }
    older_evaluation = create_model("OlderEvaluation", __config__=ConfigDict(extra="forbid"), **fields)
    task_parameters = create_model(
        "TaskParameters",
        __config__=ConfigDict(extra="forbid"),
        evaluation=(older_evaluation, Field(default_factory=older_evaluation)),
    )
    task_parameters.model_validate(_task_job_config(spec)["config"])
    explicit = SafeSynthesizerJobConfig(
        data_source=DATA_SOURCE, config={"evaluation": {"mia_excluded_columns": ["private_column"]}}
    )
    with pytest.raises(ValidationError, match="mia_excluded_columns"):
        task_parameters.model_validate(_task_job_config(explicit)["config"])


def test_explicit_unknown_field_policy_is_preserved():
    spec = SafeSynthesizerJobConfig(data_source=DATA_SOURCE, config={"unknown_fields": "ignore"})
    assert _task_job_config(spec)["config"] == {"unknown_fields": "ignore"}


def test_generated_pii_defaults_and_enable_flags_survive_reload():
    spec = SafeSynthesizerJobConfig(
        data_source=DATA_SOURCE,
        config={
            "enable_synthesis": False,
            "replace_pii": {"globals": {"classify": {"classify_model_provider": "default/model"}}},
        },
    )
    dumped = spec.model_dump(mode="json")
    assert dumped["enable_synthesis"] is False
    assert dumped["config"]["replace_pii"]["steps"]
    assert "evaluation" not in dumped["config"]
    reloaded = SafeSynthesizerJobConfig.model_validate(dumped)
    assert reloaded.model_dump(mode="json") == dumped
    assert reloaded.config.replace_pii is not None
    assert reloaded.config.replace_pii.globals.classify.classify_model_provider == "default/model"


def test_disabled_pipeline_flags_survive_storage_and_dispatch():
    spec = SafeSynthesizerJobConfig(
        data_source=DATA_SOURCE,
        config={"enable_synthesis": False, "enable_replace_pii": False},
        hf_token_secret="hf-token",
    )
    reloaded = SafeSynthesizerJobConfig.model_validate_json(spec.model_dump_json())
    dispatched = _task_job_config(reloaded)
    assert dispatched["enable_synthesis"] is False
    assert dispatched["hf_token_secret"] == "hf-token"
    assert dispatched["config"] == {"replace_pii": None}


def test_create_patch_persistence_reload_and_dispatch():
    stored = {}
    calls = []

    def response():
        value = SimpleNamespace(
            id="job-1",
            name="test-job",
            workspace="default",
            description=None,
            project=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            spec=json.loads(json.dumps(stored["spec"])),
            status="created",
            status_details={},
            error_details=None,
            ownership=None,
            custom_fields=None,
            control=None,
            attempt_id="attempt-1",
        )
        result = MagicMock()
        result.data.return_value = value
        return result

    class RecordingJobs:
        async def create_job(self, *, workspace, body):
            stored.update(json.loads(body.model_dump_json()))
            calls.append("create")
            return response()

        async def get_job(self, *, workspace, name):
            return response()

        async def update_job(self, *, workspace, name, body):
            # Description-only PATCH must not replace the saved specs.
            stored.update(json.loads(body.model_dump_json(exclude_unset=True)))
            calls.append("patch")
            return response()

    def compiler(workspace, original, transformed, entity_client, job_name, client):
        return HelixJobSpec(steps=[_create_job_step(transformed, [])])

    app = FastAPI()
    app.include_router(
        job_route_factory(
            service_name="safe-synthesizer",
            job_type="Generate",
            job_input=SafeSynthesizerJobConfig,
            platform_job_config_compiler=compiler,
        ),
        prefix="/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    client = TestClient(app)
    url = "/v2/workspaces/default/jobs"
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=RecordingJobs()):
        created = client.post(url, json={"name": "test-job", "spec": {"data_source": DATA_SOURCE, "config": {}}})
        assert created.status_code == 201, created.text
        assert stored["spec"]["config"] == {}
        assert stored["platform_spec"]["steps"][0]["config"]["config"] == {}
        assert created.json()["spec"]["config"] == {}

        explicit = {"evaluation": {"mia_excluded_columns": None}, "replace_pii": None, "unknown_fields": "reject"}
        updated = client.patch(url + "/test-job", json={"spec": {"data_source": DATA_SOURCE, "config": explicit}})
        assert updated.status_code == 200, updated.text
        assert stored["spec"]["config"] == explicit
        assert stored["platform_spec"]["steps"][0]["config"]["config"] == explicit
        assert updated.json()["spec"]["config"] == explicit
        reloaded = SafeSynthesizerJobConfig.model_validate(stored["spec"])
        assert _task_job_config(reloaded)["config"] == explicit
        unchanged = json.loads(json.dumps(stored["spec"]))
        described = client.patch(url + "/test-job", json={"description": "new description"})
        assert described.status_code == 200, described.text
        assert stored["spec"] == unchanged

        # A supplied spec replaces the definition; it is not a nested merge.
        reset = client.patch(url + "/test-job", json={"spec": {"data_source": DATA_SOURCE, "config": {}}})
        assert reset.status_code == 200, reset.text
        assert stored["spec"]["config"] == {}
    assert calls == ["create", "patch", "patch", "patch"]
