# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minikube E2E coverage for the nemo-anonymizer plugin.

These tests intentionally target an external platform deployment through
``NHX_BASE_URL``. The anonymizer.run path compiles to CPU task pods, so job
lifecycle coverage requires the K8s executor and cannot be fully validated by
the local subprocess harness.
"""

import os
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import cast

import data_designer.config as dd
import httpx
import pytest
from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, Rewrite
from anonymizer.config.replace_strategies import Hash, Redact, Substitute
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.model_configs import SelectedModelsOverrides
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest, PreviewRequest
from nemo_anonymizer_plugin.sdk.errors import (
    AnonymizerClientError,
    AnonymizerConfigValidationError,
    AnonymizerPreviewError,
)
from nemo_anonymizer_plugin.sdk.job_resources import (
    MAX_CONSECUTIVE_POLL_ERRORS,
    TERMINAL_INCOMPLETE_STATUSES,
    AnonymizerJobResource,
)
from nemo_anonymizer_plugin.sdk.resources import AnonymizerPreviewResult
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import MockProviderResponse, add_mock_provider, short_unique_name

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.timeout(1800),
]

ANONYMIZER_JOB_TIMEOUT_SECONDS = 900.0
ANONYMIZER_POLL_INTERVAL_SECONDS = 5.0
TEXT_COLUMN = "biography"
ID_COLUMN = "record_id"
DETECTED_NAME = "Alice Smith"
SUBSTITUTE_NAME = "Jordan Lee"
REPLACED_TEXT_COLUMN = f"{TEXT_COLUMN}_replaced"
REWRITTEN_TEXT_COLUMN = f"{TEXT_COLUMN}_rewritten"
CSV_REMOTE_PATH = "inputs/records.csv"
PARQUET_REMOTE_PATH = "inputs/records.parquet"
NOT_CSV_REMOTE_PATH = "inputs/records.txt"
LLM_PAYLOAD_PATH = Path(__file__).with_name("anonymizer_llm_payload.json")


def _input_rows() -> list[dict[str, str]]:
    return [
        {ID_COLUMN: "1", TEXT_COLUMN: f"{DETECTED_NAME} opened a support ticket about a router reboot."},
        {ID_COLUMN: "2", TEXT_COLUMN: f"{DETECTED_NAME} wrote a release note about a faster indexing path."},
    ]


def _input_csv() -> str:
    rows = _input_rows()
    lines = [f"{ID_COLUMN},{TEXT_COLUMN}"]
    lines.extend(f"{row[ID_COLUMN]},{row[TEXT_COLUMN]}" for row in rows)
    return "\n".join(lines) + "\n"


def _chat_completion(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-anonymizer-e2e",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _json_completion(payload: str) -> dict[str, object]:
    return _chat_completion(f"```json\n{payload}\n```")


def _detector_completion() -> dict[str, object]:
    return _chat_completion(
        '{"entities": [{"text": "Alice Smith", "label": "full_name", "start": 0, "end": 11, "score": 0.99}]}'
    )


def _llm_payload() -> str:
    return LLM_PAYLOAD_PATH.read_text(encoding="utf-8").strip()


def _string_headers(sdk: NeMoHelix) -> dict[str, str]:
    return {key: value for key, value in sdk.default_headers.items() if isinstance(value, str)}


def _workspace_client(sdk: NeMoHelix, workspace: str) -> NeMoHelix:
    return NeMoHelix(
        base_url=str(sdk.base_url).rstrip("/"),
        workspace=workspace,
        access_token=os.environ.get("NHX_ACCESS_TOKEN"),
        context_name=os.environ.get("NHX_CONTEXT_NAME"),
        max_retries=2,
        timeout=ANONYMIZER_JOB_TIMEOUT_SECONDS,
        default_headers=_string_headers(sdk),
    )


def _require_workspace(workspace: str | None) -> str:
    assert workspace is not None
    return workspace


def _anonymizer_url(sdk: NeMoHelix, workspace: str, path: str) -> str:
    return f"{str(sdk.base_url).rstrip('/')}/apis/anonymizer/v2/workspaces/{workspace}/{path.lstrip('/')}"


def _raw_anonymizer_post(
    sdk: NeMoHelix, workspace: str | None, path: str, payload: dict[str, object]
) -> httpx.Response:
    return sdk._client.post(
        _anonymizer_url(sdk, _require_workspace(workspace), path),
        json=payload,
        headers=_string_headers(sdk),
        timeout=sdk.timeout,
    )


def _assert_http_status(exc: BaseException, status_code: int) -> None:
    actual = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if actual is None and response is not None:
        actual = response.status_code
    assert actual == status_code


def _redact_config() -> AnonymizerConfig:
    return AnonymizerConfig(
        detect=Detect(entity_labels=["full_name"]),
        replace=Redact(format_template="[REDACTED_{label}]"),
        emit_telemetry=False,
    )


def _hash_config() -> AnonymizerConfig:
    return AnonymizerConfig(
        detect=Detect(entity_labels=["full_name"]),
        replace=Hash(digest_length=8),
        emit_telemetry=False,
    )


def _substitute_config() -> AnonymizerConfig:
    return AnonymizerConfig(
        detect=Detect(entity_labels=["full_name"]),
        replace=Substitute(instructions="Use harmless placeholder replacements."),
        emit_telemetry=False,
    )


def _rewrite_config() -> AnonymizerConfig:
    return AnonymizerConfig(
        detect=Detect(entity_labels=["full_name"]),
        rewrite=Rewrite(instructions="Preserve meaning while removing identifying details."),
        emit_telemetry=False,
    )


def _fileset_ref(workspace: str | None, fileset: str, path: str) -> str:
    return f"{_require_workspace(workspace)}/{fileset}#{path}"


def _fileset_uri_ref(workspace: str | None, fileset: str, path: str) -> str:
    return f"fileset://{_require_workspace(workspace)}/{fileset}#{path}"


def _input_spec(source: str, *, text_column: str = TEXT_COLUMN) -> AnonymizerInputSpec:
    return AnonymizerInputSpec(source=source, text_column=text_column, id_column=ID_COLUMN)


def _request(
    *,
    source: str,
    config: AnonymizerConfig,
    model_configs: list[dd.ModelConfig] | None,
    selected_models: SelectedModelsOverrides | None = None,
    text_column: str = TEXT_COLUMN,
) -> AnonymizerRequest:
    return AnonymizerRequest(
        config=config,
        data=_input_spec(source, text_column=text_column),
        model_configs=model_configs,
        selected_models=selected_models,
    )


def _preview_request(
    *,
    source: str,
    config: AnonymizerConfig,
    model_configs: list[dd.ModelConfig] | None,
    selected_models: SelectedModelsOverrides | None = None,
    text_column: str = TEXT_COLUMN,
    num_records: int = 2,
) -> PreviewRequest:
    request = _request(
        source=source,
        config=config,
        model_configs=model_configs,
        selected_models=selected_models,
        text_column=text_column,
    )
    return PreviewRequest(
        config=request.config,
        data=request.data,
        model_configs=request.model_configs,
        selected_models=request.selected_models,
        num_records=num_records,
    )


def _model_configs(provider_name: str) -> list[dd.ModelConfig]:
    return [
        dd.ModelConfig(alias="gliner-pii-detector", provider=provider_name, model="nvidia/gliner-pii"),
        dd.ModelConfig(alias="gpt-oss-120b", provider=provider_name, model="openai/gpt-oss-120b"),
        dd.ModelConfig(alias="nemotron-30b-thinking", provider=provider_name, model="nvidia/nemotron-3-nano-30b-a3b"),
    ]


def _mock_selected_models() -> SelectedModelsOverrides:
    return SelectedModelsOverrides(
        detection={
            "entity_detector": "gliner-pii-detector",
            "entity_validator": "gpt-oss-120b",
            "entity_augmenter": "gpt-oss-120b",
            "latent_detector": "gpt-oss-120b",
        },
        replace={"replacement_generator": "gpt-oss-120b"},
        rewrite={
            "domain_classifier": "gpt-oss-120b",
            "disposition_analyzer": "gpt-oss-120b",
            "meaning_extractor": "gpt-oss-120b",
            "qa_generator": "gpt-oss-120b",
            "rewriter": "gpt-oss-120b",
            "evaluator": "gpt-oss-120b",
            "repairer": "gpt-oss-120b",
            "judge": "gpt-oss-120b",
        },
    )


def _assert_preview_shape(result: AnonymizerPreviewResult) -> None:
    assert len(result.dataset) == 2
    assert TEXT_COLUMN in result.dataset.columns
    assert TEXT_COLUMN in result.trace_dataset.columns
    assert result.failed_records == []


def _preview_texts(result: AnonymizerPreviewResult, *, column: str = TEXT_COLUMN) -> list[str]:
    assert column in result.dataset.columns
    return [str(value) for value in result.dataset[column].tolist()]


def _assert_preview_text_changed(result: AnonymizerPreviewResult, *, column: str) -> None:
    original_texts = [row[TEXT_COLUMN] for row in _input_rows()]
    assert any(
        actual != original
        for actual, original in zip(_preview_texts(result, column=column), original_texts, strict=True)
    )


def _assert_redacted(result: AnonymizerPreviewResult) -> None:
    texts = _preview_texts(result, column=REPLACED_TEXT_COLUMN)
    assert all(DETECTED_NAME not in text for text in texts)
    assert any("[REDACTED_FULL_NAME]" in text for text in texts)


def _job_name(job: AnonymizerJobResource) -> str:
    return job._job_name  # noqa: SLF001 - SDK does not expose a public name yet.


def _wait_for_anonymizer_job(job: AnonymizerJobResource, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    status = None
    consecutive_poll_errors = 0
    last_poll_error: Exception | None = None
    while status not in {"completed", *TERMINAL_INCOMPLETE_STATUSES}:
        if time.monotonic() >= deadline:
            try:
                logs = job.get_logs()
                tail = logs[-5:] if logs else []
            except Exception as exc:
                tail = [f"<log retrieval failed: {exc!r}>"]
            raise TimeoutError(
                f"Anonymizer job {_job_name(job)} timed out with status {status!r}; "
                f"last_poll_error={last_poll_error!r}; logs={tail!r}"
            )
        try:
            status = job.get_job_status()
        except Exception as exc:
            consecutive_poll_errors += 1
            last_poll_error = exc
            if consecutive_poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise
        else:
            consecutive_poll_errors = 0
            last_poll_error = None
        if status not in {"completed", *TERMINAL_INCOMPLETE_STATUSES}:
            time.sleep(ANONYMIZER_POLL_INTERVAL_SECONDS)
    assert status == "completed"


def _cleanup_anonymizer_job(sdk: NeMoHelix, job_name: str) -> None:
    with suppress(Exception):
        jobs = client_from_platform(sdk, JobsClient)
        jobs.cancel_job(name=job_name, workspace=sdk.workspace)
    with suppress(Exception):
        jobs.delete_job(name=job_name, workspace=sdk.workspace)


@pytest.fixture(scope="module")
def anonymizer_workspace(sdk: NeMoHelix) -> Iterator[str]:
    workspaces = client_from_platform(sdk, WorkspacesClient)
    name = short_unique_name("e2e-anon")
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    try:
        yield name
    finally:
        with suppress(Exception):
            workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="module")
def anonymizer_sdk(sdk: NeMoHelix, anonymizer_workspace: str) -> Iterator[NeMoHelix]:
    client = _workspace_client(sdk, anonymizer_workspace)
    try:
        yield client
    finally:
        with suppress(Exception):
            client.close()


@pytest.fixture(scope="module")
def anonymizer_fileset(
    anonymizer_sdk: NeMoHelix,
    files_client: FilesClient,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    name = short_unique_name("anon-inputs")
    files_client.create_fileset(body=CreateFilesetRequest(name=name), workspace=anonymizer_sdk.workspace)

    files = client_from_platform(anonymizer_sdk, FilesClient)
    files.upload_file(
        name=name,
        workspace=anonymizer_sdk.workspace,
        path=CSV_REMOTE_PATH,
        content=_input_csv().encode(),
    )

    parquet_path = tmp_path_factory.mktemp("anonymizer-inputs") / "records.parquet"
    try:
        import pandas as pd

        pd.DataFrame(_input_rows()).to_parquet(parquet_path, index=False)
        files.upload_file(
            name=name,
            workspace=anonymizer_sdk.workspace,
            path=PARQUET_REMOTE_PATH,
            content=parquet_path.read_bytes(),
        )
    finally:
        with suppress(FileNotFoundError):
            parquet_path.unlink()

    files.upload_file(
        name=name,
        workspace=anonymizer_sdk.workspace,
        path=NOT_CSV_REMOTE_PATH,
        content=b"not,a,supported,input\n",
    )
    try:
        yield name
    finally:
        with suppress(Exception):
            files_client.delete_fileset(name=name, workspace=anonymizer_sdk.workspace)


@pytest.fixture(scope="module")
def mock_model_provider(sdk: NeMoHelix, anonymizer_workspace: str) -> str:
    name = short_unique_name("anon-model")
    provider = add_mock_provider(
        sdk,
        workspace=anonymizer_workspace,
        name=name,
        mock_response_body_by_model={
            "nvidia/gliner-pii": [MockProviderResponse(response_body=_detector_completion())],
            "openai/gpt-oss-120b": [MockProviderResponse(response_body=_json_completion(_llm_payload()))],
            "nvidia/nemotron-3-nano-30b-a3b": [MockProviderResponse(response_body=_json_completion(_llm_payload()))],
        },
    )
    # The module-scoped workspace deletion cascades provider/model cleanup.
    return provider.name


@pytest.fixture(scope="module")
def anonymizer_model_configs(mock_model_provider: str) -> list[dd.ModelConfig]:
    return _model_configs(mock_model_provider)


@pytest.fixture(scope="module")
def completed_redact_job(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> Iterator[AnonymizerJobResource]:
    source = _fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH)
    job = anonymizer_sdk.anonymizer.run(
        _request(
            source=source,
            config=_redact_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )
    try:
        _wait_for_anonymizer_job(job, timeout_seconds=ANONYMIZER_JOB_TIMEOUT_SECONDS)
        yield job
    finally:
        _cleanup_anonymizer_job(anonymizer_sdk, _job_name(job))


def test_health_check_through_minikube_ingress(sdk: NeMoHelix) -> None:
    response = sdk._client.get(
        f"{str(sdk.base_url).rstrip('/')}/status",
        headers=_string_headers(sdk),
    )

    response.raise_for_status()
    assert "anonymizer" in response.json()["services"]["ready"]


def test_mock_provider_chat_completion_works_through_minikube_ingress(
    sdk: NeMoHelix,
    anonymizer_workspace: str,
    mock_model_provider: str,
) -> None:
    response = sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=mock_model_provider,
        workspace=anonymizer_workspace,
        body={
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "return empty entities"}],
        },
    )

    choices = cast(list[dict[str, object]], response["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    content = message["content"]
    assert isinstance(content, str)
    assert SUBSTITUTE_NAME in content


def test_file_upload_round_trips_through_minikube_ingress(
    anonymizer_sdk: NeMoHelix, anonymizer_fileset: str
) -> None:
    content = (
        client_from_platform(anonymizer_sdk, FilesClient)
        .download_file(
            name=anonymizer_fileset,
            workspace=anonymizer_sdk.workspace,
            path=CSV_REMOTE_PATH,
        )
        .read()
    )

    assert content.decode("utf-8") == _input_csv()


def test_preview_redact_csv_happy_path(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            config=_redact_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_redacted(result)


def test_preview_hash_csv_happy_path(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            config=_hash_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_preview_text_changed(result, column=REPLACED_TEXT_COLUMN)
    assert all(DETECTED_NAME not in text for text in _preview_texts(result, column=REPLACED_TEXT_COLUMN))
    assert any("<HASH_FULL_NAME_" in text for text in _preview_texts(result, column=REPLACED_TEXT_COLUMN))


def test_preview_accepts_workspace_relative_fileset_ref(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=f"{anonymizer_fileset}#{CSV_REMOTE_PATH}",
            config=_redact_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_redacted(result)


def test_preview_accepts_fileset_uri_ref(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_uri_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            config=_redact_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_redacted(result)


def test_preview_accepts_parquet_fileset_input(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, PARQUET_REMOTE_PATH),
            config=_redact_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_redacted(result)


def test_preview_missing_text_column_is_rejected(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    request = _preview_request(
        source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
        config=_redact_config(),
        model_configs=anonymizer_model_configs,
        selected_models=_mock_selected_models(),
        text_column="missing_column",
    )

    with pytest.raises(AnonymizerPreviewError, match="missing_column"):
        anonymizer_sdk.anonymizer.preview(request)


def test_preview_invalid_strategy_payload_is_rejected(anonymizer_sdk: NeMoHelix, anonymizer_fileset: str) -> None:
    payload: dict[str, object] = {
        "config": {"replace": {"kind": "explode"}, "emit_telemetry": False},
        "data": {
            "source": _fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            "text_column": TEXT_COLUMN,
            "id_column": ID_COLUMN,
        },
        "model_configs": [],
        "num_records": 1,
    }
    response = _raw_anonymizer_post(anonymizer_sdk, anonymizer_sdk.workspace, "preview", payload)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        response.raise_for_status()
    _assert_http_status(exc_info.value, 422)


def test_preview_fileset_ref_must_point_to_file(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    with pytest.raises(AnonymizerConfigValidationError, match="#path fragment"):
        anonymizer_sdk.anonymizer.preview(
            _preview_request(
                source=f"{anonymizer_sdk.workspace}/{anonymizer_fileset}#",
                config=_redact_config(),
                model_configs=anonymizer_model_configs,
                selected_models=_mock_selected_models(),
                num_records=1,
            )
        )


def test_preview_rejects_unsupported_fileset_suffix(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    with pytest.raises(AnonymizerConfigValidationError, match=".csv or .parquet"):
        anonymizer_sdk.anonymizer.preview(
            _preview_request(
                source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, NOT_CSV_REMOTE_PATH),
                config=_redact_config(),
                model_configs=anonymizer_model_configs,
                selected_models=_mock_selected_models(),
                num_records=1,
            )
        )


def test_preview_rejects_local_path_for_remote_execution(
    anonymizer_sdk: NeMoHelix,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    missing_local_path = "./definitely-missing-local-input.csv"
    with pytest.raises(AnonymizerConfigValidationError, match="local path"):
        anonymizer_sdk.anonymizer.preview(
            _preview_request(
                source=missing_local_path,
                config=_redact_config(),
                model_configs=anonymizer_model_configs,
                selected_models=_mock_selected_models(),
                num_records=1,
            )
        )


def test_preview_selected_models_without_model_configs_is_rejected(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
) -> None:
    with pytest.raises(AnonymizerConfigValidationError, match="selected_models requires model_configs"):
        anonymizer_sdk.anonymizer.preview(
            _preview_request(
                source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
                config=_redact_config(),
                model_configs=None,
                selected_models=SelectedModelsOverrides(detection={"entity_detector": "gliner-pii-detector"}),
                num_records=1,
            )
        )


def test_preview_substitute_uses_mock_provider(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            config=_substitute_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_preview_text_changed(result, column=REPLACED_TEXT_COLUMN)
    assert all(DETECTED_NAME not in text for text in _preview_texts(result, column=REPLACED_TEXT_COLUMN))
    assert any(SUBSTITUTE_NAME in text for text in _preview_texts(result, column=REPLACED_TEXT_COLUMN))


def test_preview_rewrite_uses_mock_provider(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
    anonymizer_model_configs: list[dd.ModelConfig],
) -> None:
    result = anonymizer_sdk.anonymizer.preview(
        _preview_request(
            source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
            config=_rewrite_config(),
            model_configs=anonymizer_model_configs,
            selected_models=_mock_selected_models(),
        )
    )

    _assert_preview_shape(result)
    _assert_preview_text_changed(result, column=REWRITTEN_TEXT_COLUMN)
    assert all(DETECTED_NAME not in text for text in _preview_texts(result, column=REWRITTEN_TEXT_COLUMN))


def test_run_job_lifecycle_completes_with_k8s_executor(completed_redact_job: AnonymizerJobResource) -> None:
    assert completed_redact_job.get_job_status() == "completed"


def test_run_job_downloads_artifacts(completed_redact_job: AnonymizerJobResource, tmp_path: Path) -> None:
    completed_redact_job.download_artifacts(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    assert (artifacts_dir / "dataset.parquet").is_file()
    assert (artifacts_dir / "trace.parquet").is_file()
    assert (artifacts_dir / "metadata.json").is_file()


def test_run_job_artifacts_load_dataset_trace_and_failed_records(
    completed_redact_job: AnonymizerJobResource,
    tmp_path: Path,
) -> None:
    result = completed_redact_job.download_artifacts(tmp_path)
    dataset = result.load_dataset()
    trace = result.load_trace()

    assert len(dataset) == 2
    assert TEXT_COLUMN in dataset.columns
    assert TEXT_COLUMN in trace.columns
    assert result.load_failed_records() == []


def test_run_submit_without_model_configs_is_rejected(
    anonymizer_sdk: NeMoHelix,
    anonymizer_fileset: str,
) -> None:
    with pytest.raises(AnonymizerClientError, match="model_configs are required"):
        anonymizer_sdk.anonymizer.run(
            _request(
                source=_fileset_ref(anonymizer_sdk.workspace, anonymizer_fileset, CSV_REMOTE_PATH),
                config=_redact_config(),
                model_configs=None,
            )
        )


def test_nonexistent_run_job_returns_not_found(anonymizer_sdk: NeMoHelix) -> None:
    with pytest.raises(AnonymizerClientError) as exc_info:
        anonymizer_sdk.anonymizer.get_job_resource(short_unique_name("missing-job"))

    assert "not found" in str(exc_info.value).lower() or "404" in str(exc_info.value)
