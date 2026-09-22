# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_safe_synthesizer.config.replace_pii import ClassifyConfig, Globals, PiiReplacerConfig, StepDefinition
from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig as PluginJobConfig
from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerParameters
from nemo_safe_synthesizer_plugin.jobs import generate
from nemo_safe_synthesizer_plugin.jobs.generate import GenerateJob

DEFAULT_WORKSPACE = "default"
DEFAULT_DATA_SOURCE = "default/test-data#file.csv"


def _client_error(error_cls, status_code: int, detail: str):
    """Build a NemoHTTPError subclass from an httpx.Response, as the typed client raises."""
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(status_code=status_code, json={"detail": detail}, request=request)
    return error_cls(response)


def _patch_jobs_client(jobs_client: MagicMock, files_client: MagicMock):
    """Patch ``client_from_platform`` in the generate module, dispatching by class.

    The compiler validates the ``data_source`` fileset via
    ``client_from_platform(sdk, AsyncFilesClient).get_fileset(...)`` and then resolves
    the pretrained-model adapter via
    ``client_from_platform(sdk, AsyncJobsClient).get_job_result(...)`` — return the
    matching mock for each.
    """
    from nemo_platform_plugin.files.client import AsyncFilesClient
    from nemo_platform_plugin.jobs.client import AsyncJobsClient

    def _dispatch(_sdk, client_cls):
        if client_cls is AsyncJobsClient:
            return jobs_client
        if client_cls is AsyncFilesClient:
            return files_client
        raise AssertionError(f"unexpected client class: {client_cls!r}")

    return patch(
        "nemo_safe_synthesizer_plugin.jobs.generate.client_from_platform",
        side_effect=_dispatch,
    )


@pytest.fixture
def mock_files_client():
    mock_client = MagicMock()
    mock_client.get_fileset = AsyncMock()
    return mock_client


@pytest.fixture
def mock_sdk(mock_files_client):
    return MagicMock()


@pytest.fixture(autouse=True)
def models_client(mock_files_client):
    from nemo_platform_plugin.models.client import AsyncModelsClient

    models_client = MagicMock()
    provider_response = MagicMock()
    provider_response.data.return_value = MagicMock(name="provider")
    models_client.get_provider = AsyncMock(return_value=provider_response)
    models_client.get_provider_route_openai_url = MagicMock(
        return_value="http://nmp-host/apis/inference-gateway/v2/workspaces/default/provider/my-nim/-/v1"
    )

    def _dispatch(_sdk, client_cls):
        if client_cls is AsyncModelsClient:
            return models_client
        return mock_files_client

    with patch(
        "nemo_safe_synthesizer_plugin.jobs.generate.client_from_platform",
        side_effect=_dispatch,
    ):
        yield models_client


def _make_spec(data_source: str = DEFAULT_DATA_SOURCE, model_provider: str | None = None):
    replace_pii = None
    if model_provider is not None:
        replace_pii = PiiReplacerConfig(
            globals=Globals(classify=ClassifyConfig(classify_model_provider=model_provider)),
            steps=[StepDefinition()],
        )
    return PluginJobConfig(
        data_source=data_source,
        config=SafeSynthesizerParameters(replace_pii=replace_pii),
    )


async def _compile(spec, mock_sdk, *, profile: str | None = None, options: dict | None = None):
    return await GenerateJob.compile(
        workspace=DEFAULT_WORKSPACE,
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=mock_sdk,
        profile=profile,
        options=options,
    )


@pytest.mark.asyncio
async def test_job_config_compiler_validates_data_source(mock_sdk, mock_files_client):
    spec = _make_spec(data_source="my-workspace/my-fileset#data.csv")

    await _compile(spec, mock_sdk)

    mock_files_client.get_fileset.assert_awaited_once_with(name="my-fileset", workspace="my-workspace")


@pytest.mark.asyncio
async def test_job_config_compiler_data_source_not_found(mock_sdk, mock_files_client):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "not found"}
    mock_response.text = "not found"
    mock_files_client.get_fileset.side_effect = ClientNotFoundError(mock_response)

    with pytest.raises(PlatformJobCompilationError, match="Could not find fileset"):
        await _compile(_make_spec(), mock_sdk)


@pytest.mark.asyncio
async def test_job_config_compiler_data_source_permission_denied(mock_sdk, mock_files_client):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.json.return_value = {"detail": "denied"}
    mock_response.text = "denied"
    mock_files_client.get_fileset.side_effect = ClientPermissionDeniedError(mock_response)

    with pytest.raises(PermissionError, match="Access denied to fileset"):
        await _compile(_make_spec(), mock_sdk)


@pytest.mark.asyncio
async def test_job_config_compiler_with_classify_provider(mock_sdk, models_client):
    result = await _compile(_make_spec(model_provider="default/my-nim"), mock_sdk)

    models_client.get_provider.assert_awaited_once_with(name="my-nim", workspace="default")
    step = next(iter(result["steps"]))
    assert step["executor"]["provider"] == "gpu"
    env = {e["name"]: e.get("value") for e in step.get("environment", [])}
    assert env["CLASSIFY_LLM_ENDPOINT_PATH"] == "/apis/inference-gateway/v2/workspaces/default/provider/my-nim/-/v1"


@pytest.mark.asyncio
async def test_job_config_compiler_uses_safe_synthesizer_tasks_image(mock_sdk, monkeypatch):
    monkeypatch.setattr(generate.plugin_config, "container_image", "safe-synthesizer-tasks")
    monkeypatch.setattr(generate.plugin_config, "container_image_ref", None)
    monkeypatch.setattr(generate, "get_qualified_image", lambda name: f"registry.example.com/nemo/{name}:test-tag")

    result = await _compile(_make_spec(), mock_sdk)

    step = next(iter(result["steps"]))
    assert step["executor"]["provider"] == "gpu"
    assert step["executor"]["container"]["image"] == "registry.example.com/nemo/safe-synthesizer-tasks:test-tag"
    assert step["executor"]["container"]["entrypoint"] == [
        "python",
        "-m",
        "nemo_safe_synthesizer_plugin.tasks.safe_synthesizer",
    ]


@pytest.mark.asyncio
async def test_job_config_compiler_uses_submitted_profile(mock_sdk, monkeypatch):
    monkeypatch.setattr(generate.plugin_config, "job_executor_profile", "configured-default")

    result = await _compile(_make_spec(), mock_sdk, profile="research")

    step = next(iter(result["steps"]))
    assert step["executor"]["profile"] == "research"


@pytest.mark.asyncio
async def test_job_config_compiler_rejects_submit_options(mock_sdk):
    with pytest.raises(PlatformJobCompilationError, match="does not support submit options"):
        await _compile(_make_spec(), mock_sdk, options={"slurm": {"nodes": 4}})


@pytest.mark.asyncio
async def test_job_config_compiler_uses_image_ref_override(mock_sdk, monkeypatch):
    monkeypatch.setattr(generate.plugin_config, "container_image_ref", "safe-synthesizer-tasks:local")
    get_qualified_image = MagicMock(side_effect=AssertionError("image ref overrides should not be qualified"))
    monkeypatch.setattr(generate, "get_qualified_image", get_qualified_image)

    result = await _compile(_make_spec(), mock_sdk)

    step = next(iter(result["steps"]))
    assert step["executor"]["provider"] == "gpu"
    assert step["executor"]["container"]["image"] == "safe-synthesizer-tasks:local"
    get_qualified_image.assert_not_called()


@pytest.mark.asyncio
async def test_job_config_compiler_validates_pretrained_model_job(mock_sdk, mock_files_client):
    jobs_client = MagicMock()
    jobs_client.get_job_result = AsyncMock(
        return_value=MagicMock(artifact_url="default/job-results-prior#results/attempt-1/adapter")
    )
    spec = PluginJobConfig.model_validate(
        {
            "data_source": DEFAULT_DATA_SOURCE,
            "pretrained_model_job": "prior-safe-synth-job",
            "config": {},
        }
    )

    with _patch_jobs_client(jobs_client, mock_files_client):
        await _compile(spec, mock_sdk)

    jobs_client.get_job_result.assert_awaited_once_with(
        name="adapter",
        job="prior-safe-synth-job",
        workspace=DEFAULT_WORKSPACE,
    )


@pytest.mark.asyncio
async def test_plugin_job_config_allows_pretrained_model_job_runtime_config(mock_sdk, mock_files_client):
    jobs_client = MagicMock()
    jobs_client.get_job_result = AsyncMock(
        return_value=MagicMock(artifact_url="default/job-results-prior#results/attempt-1/adapter")
    )
    spec = PluginJobConfig.model_validate(
        {
            "data_source": DEFAULT_DATA_SOURCE,
            "pretrained_model_job": "prior-safe-synth-job",
            "config": {},
        }
    )

    with _patch_jobs_client(jobs_client, mock_files_client):
        compiled = await _compile(spec, mock_sdk)
    step = next(iter(compiled["steps"]))
    reparsed = PluginJobConfig.model_validate(step["config"])

    assert "pretrained_model" not in step["config"]["config"]["training"]
    assert reparsed.pretrained_model_job == "prior-safe-synth-job"


def test_task_job_config_allows_pretrained_model_job_with_missing_training():
    job_config = MagicMock()
    job_config.pretrained_model_job = "prior-safe-synth-job"
    job_config.model_dump.return_value = {
        "data_source": DEFAULT_DATA_SOURCE,
        "pretrained_model_job": "prior-safe-synth-job",
        "config": {"generation": {"num_records": 25}},
    }

    runtime_config = generate._task_job_config(job_config)

    assert runtime_config["config"] == {"generation": {"num_records": 25}}


def test_task_job_config_allows_pretrained_model_job_with_non_dict_training():
    job_config = MagicMock()
    job_config.pretrained_model_job = "prior-safe-synth-job"
    job_config.model_dump.return_value = {
        "data_source": DEFAULT_DATA_SOURCE,
        "pretrained_model_job": "prior-safe-synth-job",
        "config": {"training": "local-adapter"},
    }

    runtime_config = generate._task_job_config(job_config)

    assert runtime_config["config"]["training"] == "local-adapter"


def test_task_job_config_preserves_pretrained_model_without_pretrained_model_job():
    job_config = MagicMock()
    job_config.pretrained_model_job = None
    job_config.model_dump.return_value = {
        "data_source": DEFAULT_DATA_SOURCE,
        "config": {"training": {"pretrained_model": "HuggingFaceTB/SmolLM3-3B"}},
    }

    runtime_config = generate._task_job_config(job_config)

    assert runtime_config["config"]["training"]["pretrained_model"] == "HuggingFaceTB/SmolLM3-3B"


@pytest.mark.asyncio
async def test_job_config_compiler_pretrained_model_job_not_found(mock_sdk, mock_files_client):
    jobs_client = MagicMock()
    jobs_client.get_job_result = AsyncMock(side_effect=_client_error(ClientNotFoundError, 404, "not found"))
    spec = PluginJobConfig.model_validate(
        {
            "data_source": DEFAULT_DATA_SOURCE,
            "pretrained_model_job": "other-ws/prior-safe-synth-job",
            "config": {},
        }
    )

    with _patch_jobs_client(jobs_client, mock_files_client):
        with pytest.raises(PlatformJobCompilationError, match="Could not find adapter result"):
            await _compile(spec, mock_sdk)


def test_plugin_job_config_prefers_pretrained_model_job_over_conflicting_source(caplog):
    with caplog.at_level(logging.WARNING):
        result = PluginJobConfig.model_validate(
            {
                "data_source": DEFAULT_DATA_SOURCE,
                "pretrained_model_job": "prior-safe-synth-job",
                "config": {"training": {"pretrained_model": "some-other-model"}},
            }
        )

    assert result.pretrained_model_job == "prior-safe-synth-job"
    assert result.config.training.pretrained_model == "HuggingFaceTB/SmolLM3-3B"
    assert any(
        "pretrained_model_job" in record.message and "pretrained_model" in record.message for record in caplog.records
    )


def test_plugin_job_config_pretrained_model_job_matches_default_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        result = PluginJobConfig.model_validate(
            {
                "data_source": DEFAULT_DATA_SOURCE,
                "pretrained_model_job": "prior-safe-synth-job",
                "config": {"training": {"pretrained_model": "HuggingFaceTB/SmolLM3-3B"}},
            }
        )

    assert result.pretrained_model_job == "prior-safe-synth-job"
    assert result.config.training.pretrained_model == "HuggingFaceTB/SmolLM3-3B"
    assert not any(
        "pretrained_model_job" in record.message and "pretrained_model" in record.message for record in caplog.records
    )


def test_plugin_job_config_pretrained_model_job_alone_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        result = PluginJobConfig.model_validate(
            {
                "data_source": DEFAULT_DATA_SOURCE,
                "pretrained_model_job": "prior-safe-synth-job",
                "config": {},
            }
        )

    assert result.pretrained_model_job == "prior-safe-synth-job"
    assert result.config.training.pretrained_model == "HuggingFaceTB/SmolLM3-3B"
    assert not any(
        "pretrained_model_job" in record.message and "pretrained_model" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_job_config_compiler_classify_provider_wrong_format(mock_sdk):
    with pytest.raises(PlatformJobCompilationError, match="Expected 'workspace/provider_name'"):
        await _compile(_make_spec(model_provider="no-slash-here"), mock_sdk)


def test_plugin_job_config_enable_flags():
    spec = PluginJobConfig.model_validate(
        {"data_source": "default/data#file.csv", "config": {"enable_synthesis": False, "enable_replace_pii": False}}
    )

    assert spec.enable_synthesis is False
    assert spec.config.replace_pii is None


def test_plugin_job_config_enable_flags_schema():
    schema = PluginJobConfig.model_json_schema()
    config_schema = schema["$defs"]["SafeSynthesizerParameters"]

    assert "enable_synthesis" not in schema["properties"]
    assert "enable_synthesis" in config_schema["properties"]
    assert "enable_replace_pii" in config_schema["properties"]
