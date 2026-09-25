# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from nemo_helix_plugin.jobs.constants import NEMO_JOB_ID_ENVVAR, NEMO_JOB_WORKSPACE_ENVVAR


def _resp(data):
    """Wrap a payload in a NemoResponse-like object whose ``.data()`` returns it.

    Production consumes typed-client responses via ``client.<op>(...).data()``.
    """
    m = MagicMock()
    m.data.return_value = data
    return m


def import_task_main_without_heavy_runtime(monkeypatch):
    pytest.importorskip("nemo_safe_synthesizer.config.job")
    library_builder = ModuleType("nemo_safe_synthesizer.sdk.library_builder")
    setattr(library_builder, "SafeSynthesizer", MagicMock())
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.sdk.library_builder", library_builder)
    return importlib.import_module("nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.__main__")


def test_upload_results_uploads_and_registers_adapter(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)

    monkeypatch.setenv(NEMO_JOB_ID_ENVVAR, "safe-synth-job")
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "test-workspace")

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    sdk = MagicMock()
    monkeypatch.setattr(task_main, "get_platform_sdk", lambda: sdk)

    jobs_client = MagicMock()
    jobs_client.get_job.return_value = _resp(SimpleNamespace(attempt_id="attempt-123"))
    files_client = MagicMock()

    def client_from_platform(_sdk, client_cls):
        if client_cls is task_main.JobsClient:
            return jobs_client
        if client_cls is task_main.FilesClient:
            return files_client
        raise AssertionError(f"Unexpected client class: {client_cls!r}")

    monkeypatch.setattr(task_main, "client_from_platform", client_from_platform)

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

    jobs_client.get_job.assert_called_once_with(name="safe-synth-job", workspace="test-workspace")
    jobs_client.update_status_details.assert_called_once_with(
        "safe-synth-job",
        workspace="test-workspace",
        body={"output_tokens": 42},
    )

    # create_job_result is called once per result; assert the adapter call, checking the
    # artifact fields on the HelixJobResultCreateRequest body object (which replaced the
    # flat artifact_url/artifact_storage_type kwargs of the old Stainless call).
    adapter_calls = [c for c in jobs_client.create_job_result.call_args_list if c.kwargs.get("name") == "adapter"]
    assert len(adapter_calls) == 1
    adapter_call = adapter_calls[0]
    assert adapter_call.kwargs["job"] == "safe-synth-job"
    assert adapter_call.kwargs["workspace"] == "test-workspace"
    body = adapter_call.kwargs["body"]
    assert body.artifact_url == "test-workspace/job-results-safe-synth-job#results/attempt-123/adapter"
    assert body.artifact_storage_type == task_main.FileStorageType.FILESET
