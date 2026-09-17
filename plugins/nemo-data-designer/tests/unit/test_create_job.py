# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from nemo_data_designer_plugin.jobs.create import CreateJob
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig, DataDesignerStepConfig


def test_create_job_runs_step_config() -> None:
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    step_config = {
        "job_config": {"num_records": 3, "config": builder.build().model_dump()},
        "model_providers": [],
        "model_configs": [],
    }

    run_result = {
        "exit_code": 0,
        "workspace": "default",
        "num_records": 3,
        "results": {},
        "dataset_path": "file:///tmp/results/artifacts/dataset/parquet-files",
    }
    with patch(
        "nemo_data_designer_plugin.jobs.create.run_step_config_result", return_value=run_result
    ) as run_step_config_result:
        result = CreateJob().run(step_config, ctx=Mock(), sdk=Mock())

    run_step_config_result.assert_called_once()
    assert result == run_result


@pytest.mark.asyncio
async def test_validate_user_models_belong_to_accessible_providers() -> None:
    unknown_provider = "some-unknown-provider"
    bad_model_config = u.make_model_config(provider=unknown_provider)

    builder = dd.DataDesignerConfigBuilder(model_configs=[bad_model_config])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="story",
            prompt="Write a story about {{ foo }}",
            model_alias=bad_model_config.alias,
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context() as client_context,
        pytest.raises(NDDInvalidConfigError) as exc_info,
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)
    assert unknown_provider in str(exc_info.value)
    assert "Cannot access provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_user_models_are_allowed_by_providers() -> None:
    forbidden_model = "this-model-is-not-allowed"
    bad_model_config = u.make_model_config(provider=u.RESTRICTED_PROVIDER_NAME, model=forbidden_model)

    builder = dd.DataDesignerConfigBuilder(model_configs=[bad_model_config])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="story",
            prompt="Write a story about {{ foo }}",
            model_alias=bad_model_config.alias,
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
        pytest.raises(NDDInvalidConfigError) as exc_info,
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)
    assert forbidden_model in str(exc_info.value)
    assert "not enabled for provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_model_alias_values_in_column_configs() -> None:
    model_alias = "unknown-model-alias"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="school_subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="description",
            model_alias=model_alias,
            prompt="Describe the school subject {{ school_subject }}.",
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with u.make_mock_client_context(), pytest.raises(NDDInvalidConfigError) as exc_info:
        await u.compile_create_job(dd_job_config)
    assert model_alias in str(exc_info.value)
    assert "Unrecognized model alias" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_profilers() -> None:
    model_alias = "unknown-model-alias"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="school_subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    builder.add_profiler(dd.JudgeScoreProfilerConfig(model_alias=model_alias))
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with u.make_mock_client_context(), pytest.raises(NDDInvalidConfigError) as exc_info:
        await u.compile_create_job(dd_job_config)
    assert model_alias in str(exc_info.value)
    assert "Unrecognized model alias" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_hf_token_secret() -> None:
    not_a_secret_ref = "not-a-secret-ref"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(dd.HuggingFaceSeedSource(path="datasets/foo/data.parquet", token=not_a_secret_ref))
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="school_subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_secret(client_context),
        pytest.raises(NDDInvalidConfigError),
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)

    builder.with_seed_dataset(dd.HuggingFaceSeedSource(path="datasets/foo/data.parquet", token=u.SECRET_NAME))
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())
    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_secret(client_context),
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)


@pytest.mark.asyncio
async def test_validate_fileset_seed_source_is_accessible() -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(
        FilesetFileSeedSource(path="bad-workspace/bad-fileset#bad_path.parquet")  # ty: ignore[invalid-argument-type]
    )
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="school_subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context(workspace="workspace") as client_context,
        u.setup_mock_file(client_context),
        pytest.raises(NDDInvalidConfigError),
    ):
        await u.compile_create_job(dd_job_config, workspace="workspace", sdk=client_context.async_sdk)

    builder.with_seed_dataset(
        FilesetFileSeedSource(path=u.FILESET_FILE_SEED_SOURCE_PATH)  # ty: ignore[invalid-argument-type]
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())
    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_file(client_context),
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)


@pytest.mark.asyncio
async def test_successful_compilation() -> None:
    model_config = u.make_model_config(provider=u.RESTRICTED_PROVIDER_NAME, model=u.ENABLED_MODEL_NAME)
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="school_subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="description",
            model_alias=model_config.alias,
            prompt="Describe the school subject {{ school_subject }}.",
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        platform_job_spec = await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)

    platform_job_spec_dict = u._normalize_job_config(platform_job_spec)
    assert len(platform_job_spec_dict["steps"]) == 1
    platform_job_step = platform_job_spec_dict["steps"][0]

    dd_step_config = DataDesignerStepConfig(**platform_job_step["config"])
    assert dd_step_config.job_config == dd_job_config
    assert dd_step_config.model_configs == [model_config]
    assert len(dd_step_config.model_providers) == 1
    model_provider = dd_step_config.model_providers[0]
    assert model_provider.name == u.RESTRICTED_PROVIDER_NAME


@pytest.mark.asyncio
async def test_to_spec_aggregates_multiple_config_errors() -> None:
    """A single ``to_spec`` call surfaces every config problem at once."""
    # Tool configs are remote-only invalid AND the seed type is unsupported on the platform.
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[u.make_model_config()],
        tool_configs=[dd.ToolConfig(tool_alias="hello", providers=["provider"])],
    )
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=pd.DataFrame(data={"a": [1, 2, 3]})))
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
        pytest.raises(NDDInvalidConfigError) as exc_info,
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)

    msg = str(exc_info.value)
    # Both messages must surface together (no short-circuit on the first failure).
    assert "Tool configs" in msg
    assert "seed data" in msg


@pytest.mark.asyncio
async def test_to_spec_pure_internal_errors_raise_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When validate produces only internal errors, ``to_spec`` raises NDDInternalError (500 path)."""
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )
    dd_job_config = DataDesignerJobConfig(num_records=42, config=builder.build())

    async def _internal_only(self, _config):
        return [NDDInternalError("simulated internal failure")]

    monkeypatch.setattr(
        "data_designer_nemo.context.DataDesignerValidationContext.validate",
        _internal_only,
    )

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
        pytest.raises(NDDInternalError) as exc_info,
    ):
        await u.compile_create_job(dd_job_config, sdk=client_context.async_sdk)
    assert "simulated internal failure" in str(exc_info.value)
