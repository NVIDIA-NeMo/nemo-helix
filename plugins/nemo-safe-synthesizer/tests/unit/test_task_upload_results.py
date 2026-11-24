# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from nemo_helix_plugin.jobs.constants import (
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from nemo_safe_synthesizer.config.evaluation_report_data import (
    REPORT_DATA_SCHEMA_VERSION,
    EvaluationReportData,
    Metrics,
)


def import_task_main_without_heavy_runtime(monkeypatch):
    pytest.importorskip("nemo_safe_synthesizer.config.job")
    library_builder = ModuleType("nemo_safe_synthesizer.sdk.library_builder")
    setattr(library_builder, "SafeSynthesizer", MagicMock())
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.sdk.library_builder", library_builder)
    return importlib.import_module("nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.__main__")


def test_upload_results_uploads_and_registers_adapter(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)

    monkeypatch.setenv(NEMO_JOB_ID_ENVVAR, "safe-synth-job")
    monkeypatch.setenv(NEMO_JOB_ATTEMPT_ID_ENVVAR, "attempt-123")
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "test-workspace")

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    client = MagicMock()
    monkeypatch.setattr(task_main, "get_nemo_client", lambda: client)

    jobs_client = MagicMock()
    files_client = MagicMock()

    monkeypatch.setattr(task_main.JobsClient, "from_client", classmethod(lambda _cls, _client: jobs_client))
    monkeypatch.setattr(task_main.FilesClient, "from_client", classmethod(lambda _cls, _client: files_client))

    file_manager = MagicMock()
    file_manager.upload.side_effect = lambda _local_path, remote_path: (
        f"test-workspace/job-results-safe-synth-job#{remote_path}"
    )
    fileset_manager_cls = MagicMock(return_value=file_manager)
    monkeypatch.setattr(task_main, "FilesetFileManager", fileset_manager_cls)
    filesystem = MagicMock()
    fileset_filesystem_cls = MagicMock(return_value=filesystem)
    monkeypatch.setattr(task_main, "FilesetFileSystem", fileset_filesystem_cls)

    result = SimpleNamespace(
        synthetic_data=pd.DataFrame({"value": [1]}),
        summary=SimpleNamespace(
            num_completion_tokens=42,
            model_dump=lambda: {"row_count": 1, "num_completion_tokens": 42},
        ),
        evaluation_report_html=None,
        evaluation_report_data=None,
    )

    task_main.upload_results(result=result, adapter_path=adapter_path)

    fileset_manager_cls.assert_called_once_with(
        workspace="test-workspace",
        fileset_name="job-results-safe-synth-job",
        filesystem=filesystem,
        ensure_fileset_exists=True,
    )
    fileset_filesystem_cls.assert_called_once_with(client=files_client)
    file_manager.validate_storage.assert_called_once_with()
    file_manager.upload.assert_has_calls([call(adapter_path, "results/attempt-123/adapter")], any_order=True)

    jobs_client.get_job.assert_not_called()
    jobs_client.update_attempt_status_details.assert_called_once_with(
        "safe-synth-job",
        "attempt-123",
        workspace="test-workspace",
        body={"output_tokens": 42},
    )

    # create_job_result is called once per result; assert the adapter call, checking the
    # artifact fields on the HelixJobResultCreateRequest body object.
    adapter_calls = [c for c in jobs_client.create_job_result.call_args_list if c.kwargs.get("name") == "adapter"]
    assert len(adapter_calls) == 1
    adapter_call = adapter_calls[0]
    assert adapter_call.kwargs["job"] == "safe-synth-job"
    assert adapter_call.kwargs["workspace"] == "test-workspace"
    body = adapter_call.kwargs["body"]
    assert body.attempt_id == "attempt-123"
    assert body.artifact_url == "test-workspace/job-results-safe-synth-job#results/attempt-123/adapter"
    assert body.artifact_storage_type == task_main.FileStorageType.FILESET


def _patch_upload_dependencies(task_main, monkeypatch):
    monkeypatch.setenv(NEMO_JOB_ID_ENVVAR, "safe-synth-job")
    monkeypatch.setenv(NEMO_JOB_ATTEMPT_ID_ENVVAR, "attempt-123")
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "test-workspace")
    monkeypatch.setattr(task_main, "get_nemo_client", MagicMock)
    jobs_client = MagicMock()
    monkeypatch.setattr(task_main.JobsClient, "from_client", classmethod(lambda _cls, _client: jobs_client))
    monkeypatch.setattr(task_main.FilesClient, "from_client", classmethod(lambda _cls, _client: MagicMock()))
    monkeypatch.setattr(task_main, "FilesetFileSystem", MagicMock())

    uploaded: dict[str, str] = {}

    def upload(local_path, remote_path):
        uploaded[remote_path] = local_path.read_text(encoding="utf-8") if local_path.is_file() else ""
        return f"test-workspace/job-results-safe-synth-job#{remote_path}"

    file_manager = MagicMock()
    file_manager.upload.side_effect = upload
    monkeypatch.setattr(task_main, "FilesetFileManager", MagicMock(return_value=file_manager))
    return jobs_client, uploaded


def _report_data() -> EvaluationReportData:
    return EvaluationReportData.model_validate(
        {
            **dict.fromkeys(EvaluationReportData.model_fields),
            "schema_version": REPORT_DATA_SCHEMA_VERSION,
            "library_version": "test",
            "job_id": "safe-synth-job",
            "scores": {"synthetic_quality": None, "data_privacy": None},
            "metrics": dict.fromkeys(Metrics.model_fields),
            "columns": [],
            "differential_privacy": {"enabled": False, "epsilon": None, "delta": None},
        }
    )


def _evaluated_result(report_data: EvaluationReportData | None):
    return SimpleNamespace(
        synthetic_data=pd.DataFrame({"value": [1]}),
        summary=SimpleNamespace(num_completion_tokens=None, model_dump=lambda: {"row_count": 1}),
        evaluation_report_html="<html></html>",
        evaluation_report_data=report_data,
    )


def _created_result_names(jobs_client) -> list[str]:
    return [c.kwargs["name"] for c in jobs_client.create_job_result.call_args_list]


def test_upload_results_uploads_report_data(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    jobs_client, uploaded = _patch_upload_dependencies(task_main, monkeypatch)
    report_data = _report_data()

    task_main.upload_results(result=_evaluated_result(report_data))

    assert _created_result_names(jobs_client) == [
        "synthetic-data",
        "summary",
        "evaluation-report",
        "evaluation-report-data",
    ]
    assert json.loads(uploaded["results/attempt-123/evaluation-report-data"]) == report_data.model_dump(mode="json")
    report_data_body = jobs_client.create_job_result.call_args_list[-1].kwargs["body"]
    assert report_data_body.artifact_url == (
        "test-workspace/job-results-safe-synth-job#results/attempt-123/evaluation-report-data"
    )


def test_upload_results_skips_report_data_when_the_library_has_none(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    jobs_client, uploaded = _patch_upload_dependencies(task_main, monkeypatch)

    task_main.upload_results(result=_evaluated_result(None))

    assert _created_result_names(jobs_client) == ["synthetic-data", "summary", "evaluation-report"]
    assert "results/attempt-123/evaluation-report-data" not in uploaded


def test_upload_results_keeps_the_adapter_when_the_report_data_upload_fails(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    jobs_client, uploaded = _patch_upload_dependencies(task_main, monkeypatch)
    file_manager = task_main.FilesetFileManager.return_value
    upload = file_manager.upload.side_effect

    def upload_failing_for_report_data(local_path, remote_path):
        if remote_path.endswith("evaluation-report-data"):
            raise OSError("files service unavailable")
        return upload(local_path, remote_path)

    file_manager.upload.side_effect = upload_failing_for_report_data
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    task_main.upload_results(result=_evaluated_result(_report_data()), adapter_path=adapter_path)

    assert _created_result_names(jobs_client) == ["synthetic-data", "summary", "evaluation-report", "adapter"]
    assert "results/attempt-123/evaluation-report-data" not in uploaded
