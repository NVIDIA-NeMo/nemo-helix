# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from data_designer.errors import DataDesignerError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from filesets import transfer
from nemo_data_designer_plugin.config import get_config
from nemo_data_designer_plugin.sdk.errors import DataDesignerConfigValidationError, DataDesignerPreviewError
from nemo_data_designer_plugin.sdk.resources import DataDesignerResource
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.testing import ClientContext

pytestmark = pytest.mark.integration


def test_request_too_many_records(data_designer: DataDesignerResource) -> None:
    too_many_records = get_config().preview_num_records.max + 1

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a", "b"]),
        )
    )

    with pytest.raises(DataDesignerConfigValidationError) as exc_info:
        data_designer.preview(builder, num_records=too_many_records)
    assert "Max num records" in str(exc_info.value)


def test_happy_path_preview(data_designer: DataDesignerResource) -> None:
    column_name = "column-name"
    value = "a"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name=column_name,
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[value]),
        )
    )

    with capture_sdk_preview_log_messages() as log_messages:
        preview_results = data_designer.preview(builder, num_records=3)

    expected_dataset = pd.DataFrame(data={column_name: [value, value, value]}).convert_dtypes(dtype_backend="pyarrow")
    assert preview_results.dataset is not None
    pd.testing.assert_frame_equal(preview_results.dataset, expected_dataset)
    assert preview_results.dataset_metadata is not None
    assert preview_results.analysis is not None

    assert_message_with(log_messages, fuzzy=column_name)
    assert_message_with(log_messages, fuzzy="Preview generation in progress")


@pytest.mark.usefixtures("mock_secret")
def test_hf_seed_dataset(data_designer: DataDesignerResource) -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(
        dd.HuggingFaceSeedSource(path="my-ws/my-fileset#path/to/data.parquet", token=u.SECRET_NAME)
    )
    builder.add_column(column_config=dd.ExpressionColumnConfig(name="full_name", expr=u.FULL_NAME_EXPR))

    with u.mock_hf_seed_reader():
        preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert set(preview_results.dataset["full_name"].values) == u.FULL_NAMES


@pytest.mark.usefixtures("mock_file")
def test_fileset_file_seed_dataset_plugin(data_designer: DataDesignerResource) -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(FilesetFileSeedSource(path=u.FILESET_FILE_SEED_SOURCE_PATH))  # ty: ignore[invalid-argument-type]
    builder.add_column(column_config=dd.ExpressionColumnConfig(name="full_name", expr=u.FULL_NAME_EXPR))

    preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert set(preview_results.dataset["full_name"].values) == u.FULL_NAMES


@pytest.mark.usefixtures("mock_file")
def test_directory_seed_dataset_fileset_root(data_designer: DataDesignerResource) -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(dd.DirectorySeedSource(path=f"{u.WORKSPACE_NAME}/{u.FILESET_NAME}"))
    builder.add_column(
        column_config=dd.ExpressionColumnConfig(
            name="expr",
            expr="{{ source_kind }} :: {{ source_path }} :: {{ relative_path }}",
        )
    )

    preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert set(preview_results.dataset["expr"].values) == {
        f"directory_file :: {u.WORKSPACE_NAME}/{u.FILESET_NAME}#{u.FILE_PATH} :: {u.FILE_PATH}"
    }


def test_directory_seed_dataset_fileset_subdir(
    client_context: ClientContext, data_designer: DataDesignerResource
) -> None:
    subdir = "some/subdir"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(dd.DirectorySeedSource(path=f"{u.WORKSPACE_NAME}/{u.FILESET_NAME}#{subdir}"))
    builder.add_column(
        column_config=dd.ExpressionColumnConfig(
            name="expr",
            expr="{{ source_kind }} :: {{ relative_path }}",
        )
    )

    with u.setup_mock_file(client_context, remote_path=f"{subdir}/{u.FILE_PATH}"):
        preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert set(preview_results.dataset["expr"].values) == {f"directory_file :: {u.FILE_PATH}"}


def test_file_contents_seed_dataset(files_client: FilesClient, data_designer: DataDesignerResource) -> None:
    subdir = "some/subdir"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.with_seed_dataset(dd.FileContentsSeedSource(path=f"{u.WORKSPACE_NAME}/{u.FILESET_NAME}#{subdir}"))
    builder.add_column(
        column_config=dd.ExpressionColumnConfig(
            name="expr",
            expr="{{ file_name }} :: {{ content }}",
        )
    )

    files_client.create_fileset(body=CreateFilesetRequest(name=u.FILESET_NAME), workspace=u.WORKSPACE_NAME)
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in ["abc.txt", "xyz.txt"]:
            filepath = Path(tmpdir) / filename
            filepath.write_text(f"This is {filename}")
        transfer.upload(
            files_client,
            fileset=u.FILESET_NAME,
            workspace=u.WORKSPACE_NAME,
            local_path=tmpdir,
            remote_path=subdir,
        )

    preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert set(preview_results.dataset["expr"].values) == {
        "abc.txt :: This is abc.txt",
        "xyz.txt :: This is xyz.txt",
    }


def test_bad_config_source_shows_clear_error_and_no_traceback(tmp_path: Path) -> None:
    config_path = u.write_config_file(
        tmp_path,
        """
import data_designer.config as dd
import pandas as pd


def wrong_function_name() -> dd.DataDesignerConfigBuilder:
    return dd.DataDesignerConfigBuilder()
""",
        name="dataframe_seed_create_config.py",
    )

    with u.make_mock_client_context(workspace="default") as client_context:
        result = u.invoke_cli(
            ["create", str(config_path), "--num-records", "3"],
            client_context,
        )

    assert result.exit_code != 0
    assert "load_config_builder()" in result.output
    assert "traceback" not in result.output.lower()


def test_nemotron_personas_dataset(client_context: ClientContext, data_designer: DataDesignerResource) -> None:
    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="customer",
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(locale="en_US", age_range=[25, 45]),
        )
    )
    builder.add_column(column_config=dd.ExpressionColumnConfig(name="demo", expr="Customer is {{ customer.age }}"))

    sample_persona_data = pd.DataFrame(
        data={"first_name": ["Charlie"] * 100, "last_name": ["Parker"] * 100, "age": list(range(100))}
    )

    with u.setup_mock_nemotron_personas_data(client_context, sample_persona_data):
        preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None

    def _parse_age(v: str) -> int:
        return int(v.removeprefix("Customer is "))

    demo_values = preview_results.dataset["demo"].values
    demo_ages = [_parse_age(value) for value in demo_values]
    assert all(25 <= age <= 45 for age in demo_ages)


def test_preview_with_schema_transform_processor(data_designer: DataDesignerResource) -> None:
    column_name = "school_subject"
    processor_name = "chat_format"

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name=column_name,
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math", "science", "history"]),
        )
    )
    builder.add_processor(
        dd.SchemaTransformProcessorConfig(
            name=processor_name,
            template={
                "messages": [
                    {"role": "user", "content": "{{school_subject}}"},
                    {"role": "assistant", "content": "{{school_subject}}"},
                ],
            },
        )
    )

    preview_results = data_designer.preview(builder, num_records=3)

    assert preview_results.dataset is not None
    assert preview_results.processor_artifacts is not None
    assert processor_name in preview_results.processor_artifacts
    processor_records = preview_results.processor_artifacts[processor_name]
    assert isinstance(processor_records, list)
    assert len(processor_records) == 3
    assert "messages" in processor_records[0]


def test_preview_surfaces_worker_error_through_sdk(
    monkeypatch: pytest.MonkeyPatch, data_designer: DataDesignerResource
) -> None:
    """When the preview worker thread raises, the function emits a ``LogFrame`` and an
    ``Error`` frame instead of ``Done``; the SDK's ``_PreviewFrameCollector`` translates
    that ``Error`` into a typed ``DataDesignerPreviewError`` with the original message.
    """
    from nemo_data_designer_plugin.functions import preview as preview_module

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced worker failure")

    monkeypatch.setattr(preview_module, "_make_preview_dataset", boom)

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a"]),
        )
    )

    with pytest.raises(DataDesignerPreviewError, match="forced worker failure"):
        data_designer.preview(builder, num_records=3)


def test_preview_model_health_check_failure_returns_validation_error(
    monkeypatch: pytest.MonkeyPatch, data_designer: DataDesignerResource
) -> None:
    # It would be preferable to set up a mock provider configured to return a 5xx response,
    # which is supported in the shared test helper, but doesn't work here because the call
    # to the provider comes from the Data Designer library's own client, which does not carry
    # the ASGI transport configured in the tests. So instead, we patch `check_models` to raise.
    error_message = "forced model health check failure"

    def fail_model_health_check(*args, **kwargs) -> None:
        raise DataDesignerError(error_message)

    monkeypatch.setattr(DataDesigner, "check_models", fail_model_health_check)

    builder = dd.DataDesignerConfigBuilder(model_configs=[u.make_model_config()])
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a"]),
        )
    )

    with pytest.raises(DataDesignerConfigValidationError, match=error_message) as exc_info:
        data_designer.preview(builder, num_records=3)

    assert exc_info.value.status_code == 422


def assert_message_with(messages: list[str], exact: str | None = None, fuzzy: str | None = None) -> None:
    match (exact, fuzzy):
        case (None, None):
            raise ValueError("Must set 'exact' OR 'fuzzy', not both")
        case (some_exact, None):
            assert some_exact in messages
        case (None, some_fuzzy):
            assert any(some_fuzzy in message for message in messages), (
                f"Did not find a message with expected content: {some_fuzzy}"
            )
        case (_, _):
            raise ValueError("Must set 'exact' OR 'fuzzy', not both")


class _MessageCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_sdk_preview_log_messages() -> Iterator[list[str]]:
    preview_logger = logging.getLogger("nemo_data_designer_plugin.sdk.resources")
    previous_level = preview_logger.level
    handler = _MessageCaptureHandler()

    preview_logger.addHandler(handler)
    preview_logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        preview_logger.removeHandler(handler)
        preview_logger.setLevel(previous_level)
