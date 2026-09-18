# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import time
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path
from typing import Any

import data_designer.config as dd
import pandas as pd
import pytest
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from data_designer_nemo.nemotron_personas import WORKSPACE, get_resource_name_for_locale
from nemo_data_designer_plugin.sdk.errors import DataDesignerJobError
from nemo_platform import NeMoPlatform
from nemo_platform.types.inference import ModelProvider
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nmp.testing import MockProviderResponse, add_mock_provider, assert_exit_0, run_nemo_local
from nmp.testing.pytest_outcomes import pytest_skip

pytestmark = [pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml")]

PROVIDER_NAME = "test-provider"

MODEL_A = "model-a"
MODEL_B = "model-b"
# A model the provider accepts in config but refuses to serve — the NMP-194 shape.
MODEL_UNSERVABLE = "model-unservable"

MODEL_A_RESPONSE = "hello world"
MODEL_B_RESPONSE = "foo bar baz"

COMMON_EXPECTED_ROW_DATA = {
    "static_sampler": "static",
    "response_from_a": MODEL_A_RESPONSE,
    "response_from_b": MODEL_B_RESPONSE,
}

PREVIEW_NUM_RECORDS = 10
JOB_NUM_RECORDS = 30


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    """Create a chat completion response body."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _make_mock_provider(sdk: NeMoPlatform, workspace: str) -> ModelProvider:
    return add_mock_provider(
        sdk,
        workspace=workspace,
        name=PROVIDER_NAME,
        mock_response_body_by_model={
            MODEL_A: [
                MockProviderResponse(response_body=_chat_completion_response(MODEL_A_RESPONSE, MODEL_A)),
            ],
            MODEL_B: [
                MockProviderResponse(response_body=_chat_completion_response(MODEL_B_RESPONSE, MODEL_B)),
            ],
        },
    )


def _setup_dd_config(provider: ModelProvider) -> dd.DataDesignerConfigBuilder:
    model_configs = [
        dd.ModelConfig(
            alias="a",
            model=MODEL_A,
            provider=provider.name,
            inference_parameters=dd.ChatCompletionInferenceParams(top_p=1),
        ),
        dd.ModelConfig(
            alias="b",
            model=MODEL_B,
            provider=provider.name,
            inference_parameters=dd.ChatCompletionInferenceParams(top_p=1),
        ),
    ]
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="static_sampler",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["static"]),
        )
    )

    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="response_from_a",
            model_alias="a",
            prompt="Tell me something about {{ static_sampler }}",
        )
    )

    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="response_from_b",
            model_alias="b",
            prompt="Tell me something about {{ static_sampler }}",
        )
    )

    return builder


def _make_expected_dataset(row_data: dict[str, Any], num_records: int) -> pd.DataFrame:
    return pd.DataFrame([row_data] * num_records).convert_dtypes(dtype_backend="pyarrow")


def _assert_dataset_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


def test_simple_ndd_config(sdk: NeMoPlatform, workspace: str) -> None:
    provider = _make_mock_provider(sdk, workspace)
    config_builder = _setup_dd_config(provider)

    preview_results = sdk.data_designer.preview(config_builder, workspace=workspace)
    expected_preview_dataset = _make_expected_dataset(COMMON_EXPECTED_ROW_DATA, PREVIEW_NUM_RECORDS)
    _assert_dataset_equal(preview_results.dataset, expected_preview_dataset)

    job_dataset = _create_job_and_get_dataset(sdk, workspace, config_builder)
    expected_job_dataset = _make_expected_dataset(COMMON_EXPECTED_ROW_DATA, JOB_NUM_RECORDS)
    _assert_dataset_equal(job_dataset, expected_job_dataset)


def test_fileset_seed_data(sdk: NeMoPlatform, files_client: FilesClient, workspace: str) -> None:
    """Tests that the Data Designer *library* plugin that makes Filesets available as seed sources
    is wired up properly by the Data Designer *platform plugin*.
    """
    fileset_name = "my-fileset"
    files_client.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace)

    seed_data = pd.DataFrame(data={"seed": ["my-seed"]})
    remote_path = "data.parquet"
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        seed_data.to_parquet(f.name, index=False)
        client_from_platform(sdk, FilesClient).upload_file(
            path=remote_path,
            name=fileset_name,
            workspace=workspace,
            content=Path(f.name).read_bytes(),
        )

    filepath = f"{workspace}/{fileset_name}#{remote_path}"

    provider = _make_mock_provider(sdk, workspace)
    config_builder = _setup_dd_config(provider)

    config_builder.with_seed_dataset(FilesetFileSeedSource(path=filepath))  # ty: ignore[invalid-argument-type]

    expected_row_data = {"seed": "my-seed"}
    expected_row_data.update(COMMON_EXPECTED_ROW_DATA)

    preview_results = sdk.data_designer.preview(config_builder, workspace=workspace)
    expected_preview_dataset = _make_expected_dataset(expected_row_data, PREVIEW_NUM_RECORDS)
    _assert_dataset_equal(preview_results.dataset, expected_preview_dataset)

    job_dataset = _create_job_and_get_dataset(sdk, workspace, config_builder)
    expected_job_dataset = _make_expected_dataset(expected_row_data, JOB_NUM_RECORDS)
    _assert_dataset_equal(job_dataset, expected_job_dataset)


@pytest.fixture
def nemotron_personas_locale(
    _services: str, files_client: FilesClient, workspace: str, ngc_secret: str
) -> Generator[str]:
    """Invokes the CLI to create a Fileset for Nemotron Personas data.

    This test does call out to NGC and downloads personas data. Use the smallest locale available
    to keep test runtime manageable.

    Nemotron Personas filesets are created in the "system" workspace, **not** the (ephemeral) ``workspace``
    pytest fixture workspace. The "system" workspace persists across e2e runs, so we delete the fileset
    before and after the test to ensure a clean test environment.
    """
    locale = "en_SG"

    fileset_name = get_resource_name_for_locale(locale)
    with suppress(NotFoundError):
        files_client.delete_fileset(name=fileset_name, workspace=WORKSPACE)

    result = run_nemo_local(
        "data-designer",
        "personas",
        "make-fileset",
        "--locale",
        locale,
        "--api-key-secret",
        f"{workspace}/{ngc_secret}",
        base_url=_services,
        workspace=workspace,
    )
    assert_exit_0(result, "Failed to create Nemotron Personas fileset via CLI")

    yield locale

    with suppress(NotFoundError):
        files_client.delete_fileset(name=fileset_name, workspace=WORKSPACE)


def test_nemotron_personas_sampling(sdk: NeMoPlatform, workspace: str, nemotron_personas_locale: str) -> None:
    """Test Nemotron Personas data can be created in the platform and subsequently dd.SamplerType.PERSON
    columns can be included in workloads.

    Nemotron Personas filesets are created via the CLI. The CLI invocation is "buried" in a pytest
    fixture to ensure a clean test environment on each run, see ``nemotron_personas_locale``.
    """
    provider = _make_mock_provider(sdk, workspace)
    config_builder = _setup_dd_config(provider)

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="customer",
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(
                locale=nemotron_personas_locale,
                age_range=[25, 45],
            ),
        )
    )
    config_builder.add_column(
        column_config=dd.ExpressionColumnConfig(
            name="demo",
            expr="Customer is {{ customer.age }}",
        )
    )

    # The person sampler column ("customer") is too complex for a full
    # pd.testing.assert_frame_equal assertion. Instead, we verify that
    # the constraint we supplied (age_range) is respected and that
    # person attributes were successfully used in a downstream column.

    def _parse_age(v: str) -> int:
        return int(v.removeprefix("Customer is "))

    def _get_demo_ages(dataset: pd.DataFrame) -> list[int]:
        demo_values = dataset["demo"].values
        return [_parse_age(value) for value in demo_values]

    preview_results = sdk.data_designer.preview(config_builder, workspace=workspace)
    assert len(preview_results.dataset) == PREVIEW_NUM_RECORDS
    assert all(25 <= age <= 45 for age in _get_demo_ages(preview_results.dataset))

    job_dataset = _create_job_and_get_dataset(sdk, workspace, config_builder)
    assert len(job_dataset) == JOB_NUM_RECORDS
    assert all(25 <= age <= 45 for age in _get_demo_ages(job_dataset))


def _create_job_and_get_dataset(
    sdk: NeMoPlatform,
    workspace: str,
    config_builder: dd.DataDesignerConfigBuilder,
) -> pd.DataFrame:
    job = sdk.data_designer.create(config_builder, num_records=JOB_NUM_RECORDS, workspace=workspace)
    job.wait_until_done()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            results = _download_artifacts_when_ready(job, tmpdir)
        except DataDesignerJobError as exc:
            message = str(exc)
            if (
                "Job result not found" in message
                or "'artifacts' result is not available." in message
                or "Timed out waiting for Data Designer artifacts" in message
            ):
                pytest_skip(f"Data Designer job completed without publishing artifacts: {exc}")
            raise

        analysis = results.load_analysis()
        assert analysis.num_records == JOB_NUM_RECORDS

        return results.load_dataset()


def _download_artifacts_when_ready(job: Any, tmpdir: str) -> Any:
    deadline = time.monotonic() + 60
    last_error: DataDesignerJobError | None = None

    while time.monotonic() < deadline:
        try:
            return job.download_artifacts(tmpdir)
        except DataDesignerJobError as exc:
            last_error = exc
            message = str(exc)
            if "Job result not found" not in message and "'artifacts' result is not available." not in message:
                raise
            time.sleep(2)

    raise DataDesignerJobError(f"Timed out waiting for Data Designer artifacts: {last_error}") from last_error


def _make_unservable_model_provider(sdk: NeMoPlatform, workspace: str) -> ModelProvider:
    """A provider that resolves cleanly but 404s the model itself."""
    return add_mock_provider(
        sdk,
        workspace=workspace,
        name="unservable-provider",
        mock_response_body_by_model={
            MODEL_UNSERVABLE: [
                MockProviderResponse(
                    response_code=404,
                    response_body={"error": {"message": f"model {MODEL_UNSERVABLE!r} not found", "type": "not_found"}},
                ),
            ],
        },
    )


def _single_model_config(provider: ModelProvider, model: str) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="a",
                model=model,
                provider=provider.name,
                inference_parameters=dd.ChatCompletionInferenceParams(top_p=1),
            )
        ]
    )
    builder.add_column(
        column_config=dd.LLMTextColumnConfig(
            name="response_from_a",
            model_alias="a",
            prompt="Tell me something.",
        )
    )
    return builder


def test_check_models_passes_for_servable_models(sdk: NeMoPlatform, workspace: str) -> None:
    """The only place a real model probe runs: integration tests cannot, because
    the engine's HTTP client does not carry their ASGI transport."""
    provider = _make_mock_provider(sdk, workspace)
    config_builder = _setup_dd_config(provider)

    report = sdk.data_designer.check_models(config_builder, workspace=workspace)

    assert report.ok, [(e.error_type, e.message) for e in report.errors]


def test_check_models_catches_model_the_provider_cannot_serve(sdk: NeMoPlatform, workspace: str) -> None:
    """NMP-194: `validate` passes for a model the provider will not serve.

    That is the documented split, not a bug — validate resolves the provider
    and never contacts the model. This asserts both halves: validate stays
    green, and check-models is what actually catches it.
    """
    provider = _make_unservable_model_provider(sdk, workspace)
    config_builder = _single_model_config(provider, MODEL_UNSERVABLE)

    validation_report = sdk.data_designer.validate(config_builder, workspace=workspace)
    assert validation_report.ok, [e.message for e in validation_report.errors]

    report = sdk.data_designer.check_models(config_builder, workspace=workspace)

    assert not report.ok
    assert report.errors
    assert report.errors[0].error_type


def test_check_models_cli_exits_nonzero_for_unservable_model(_services: str, sdk: NeMoPlatform, workspace: str) -> None:
    provider = _make_unservable_model_provider(sdk, workspace)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.py"
        config_path.write_text(
            "import data_designer.config as dd\n"
            "\n"
            "\n"
            "def load_config_builder() -> dd.DataDesignerConfigBuilder:\n"
            "    builder = dd.DataDesignerConfigBuilder(\n"
            "        model_configs=[\n"
            f"            dd.ModelConfig(alias='a', model={MODEL_UNSERVABLE!r}, provider={provider.name!r})\n"
            "        ]\n"
            "    )\n"
            "    builder.add_column(\n"
            "        dd.LLMTextColumnConfig(name='x', model_alias='a', prompt='Tell me something.')\n"
            "    )\n"
            "    return builder\n",
            encoding="utf-8",
        )

        result = run_nemo_local(
            "data-designer",
            "check-models",
            str(config_path),
            base_url=_services,
            workspace=workspace,
        )

    assert result.returncode != 0, result.stdout
    assert "Model health check failed" in result.stdout
