# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared customization file_io runner."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import pytest


def _make_job_ctx(*, workspace: str = "default", storage_path: Path | None = None):
    from nmp.customization_common.service.context import NMPJobContext

    return NMPJobContext(
        workspace=workspace,
        job_id="job-1",
        attempt_id="attempt-0",
        step="model-upload",
        task="task-1",
        jobs_url=None,
        files_url=None,
        storage_path=storage_path or Path("/tmp"),
        config_path=Path("/tmp/cfg.json"),
    )


def _make_runner(
    files,
    *,
    service_source: str = "unsloth",
    workspace: str = "default",
    storage_path: Path | None = None,
):
    from nmp.customization_common.tasks.file_io.run import FileIORunner
    from nmp.customization_common.tasks.file_io_progress_reporter import NoOpProgressReporter

    job_ctx = _make_job_ctx(workspace=workspace, storage_path=storage_path)
    return FileIORunner(
        files=files,
        progress_reporter=NoOpProgressReporter(),
        job_ctx=job_ctx,
        service_source=service_source,
    )


def _make_files_client() -> MagicMock:
    files = MagicMock()
    files.with_options.return_value = files
    return files


def _raise_runner_conflict() -> None:
    import sys

    run_mod = sys.modules["nmp.customization_common.tasks.file_io.run"]
    raise run_mod.ConflictError.__new__(run_mod.ConflictError, "already exists")


def _make_dir(tmp_path: Path) -> Path:
    src = tmp_path / "checkpoint"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    (src / "tokenizer.json").write_text("{}")
    return src


class TestCreateFileset:
    def test_creates_fileset_with_service_source_and_metadata(self) -> None:
        from nemo_platform_plugin.files.types import CreateFilesetRequest
        from nmp.customization_common.schemas.file_io import FileSetRef

        files = _make_files_client()
        runner = _make_runner(files, service_source="automodel")
        metadata = {"model": {"tool_calling": {"tool_call_parser": "llama3_json"}}}
        dest = FileSetRef(workspace="default", name="qwen-test")

        runner.create_fileset(dest, metadata=metadata)

        files.create_fileset.assert_called_once()
        call = files.create_fileset.call_args
        assert call.kwargs["workspace"] == "default"
        body = call.kwargs["body"]
        assert isinstance(body, CreateFilesetRequest)
        assert body.name == "qwen-test"
        assert body.custom_fields == {"service_source": "automodel"}
        assert body.metadata is not None
        assert body.metadata.model is not None
        assert body.metadata.model.tool_calling.tool_call_parser == "llama3_json"

    def test_conflict_patches_metadata_on_existing(self) -> None:
        from nemo_platform_plugin.files.types import UpdateFilesetRequest
        from nmp.customization_common.schemas.file_io import FileSetRef

        files = _make_files_client()
        files.create_fileset.side_effect = lambda **_: _raise_runner_conflict()
        runner = _make_runner(files, service_source="rl")
        dest = FileSetRef(workspace="default", name="exists")
        metadata = {"model": {"tool_calling": {"tool_call_parser": "hermes"}}}

        runner.create_fileset(dest, metadata=metadata)

        files.update_fileset.assert_called_once()
        update_call = files.update_fileset.call_args
        assert update_call.kwargs["workspace"] == "default"
        assert update_call.kwargs["name"] == "exists"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateFilesetRequest)
        assert body.metadata is not None
        assert body.metadata.model is not None
        assert body.metadata.model.tool_calling.tool_call_parser == "hermes"

    def test_conflict_no_metadata_skips_update(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef

        files = _make_files_client()
        files.create_fileset.side_effect = lambda **_: _raise_runner_conflict()
        runner = _make_runner(files)
        dest = FileSetRef(workspace="default", name="exists")

        runner.create_fileset(dest, metadata=None)

        files.update_fileset.assert_not_called()


class TestUploadFileset:
    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_uses_files_client_for_transfer(self, mock_fs_cls, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.tasks.file_io.run import UPLOAD_TIMEOUT

        files = _make_files_client()
        fs = MagicMock()
        mock_fs_cls.return_value = fs
        runner = _make_runner(files)
        src = _make_dir(tmp_path)

        runner.upload_fileset(FileSetRef(workspace="default", name="qwen-test"), src.resolve())

        files.with_options.assert_called_once_with(timeout=UPLOAD_TIMEOUT)
        mock_fs_cls.assert_called_once_with(client=files)

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_directory_uploads_with_trailing_slash(self, mock_fs_cls, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef

        fs = MagicMock()
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())
        src = _make_dir(tmp_path)
        dest = FileSetRef(workspace="default", name="qwen-test")

        runner.upload_fileset(dest, src.resolve())

        fs.put.assert_called_once()
        put_call = fs.put.call_args
        assert put_call.args[0] == f"{src.resolve()}/"
        assert put_call.args[1] == "default/qwen-test"
        assert put_call.kwargs["recursive"] is True

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_upload_failure_propagates_as_file_upload_error(self, mock_fs_cls, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef, FileUploadError

        fs = MagicMock()
        fs.put.side_effect = RuntimeError("upload broke")
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())
        src = _make_dir(tmp_path)
        dest = FileSetRef(workspace="default", name="x")

        with pytest.raises(FileUploadError, match="upload broke"):
            runner.upload_fileset(dest, src.resolve())


class TestDownloadFileset:
    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_uses_files_client_for_transfer(self, mock_fs_cls, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.tasks.file_io.run import DOWNLOAD_TIMEOUT, LIST_FILES_TIMEOUT

        fs = MagicMock()
        mock_fs_cls.return_value = fs
        files = _make_files_client()
        files.list_files.return_value.data.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(path="model.safetensors", size=100)]
        )
        runner = _make_runner(files)

        runner.download_fileset(FileSetRef(workspace="default", name="qwen"), tmp_path / "downloads")

        files.with_options.assert_has_calls([call(timeout=LIST_FILES_TIMEOUT), call(timeout=DOWNLOAD_TIMEOUT)])
        mock_fs_cls.assert_called_once_with(client=files)

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_lists_then_downloads(self, mock_fs_cls, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef

        fs = MagicMock()
        mock_fs_cls.return_value = fs
        files = _make_files_client()
        files.list_files.return_value.data.return_value = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(path="model.safetensors", size=100),
                types.SimpleNamespace(path="config.json", size=20),
            ]
        )
        runner = _make_runner(files)
        dest = tmp_path / "downloads"
        fileset = FileSetRef(workspace="default", name="qwen")

        runner.download_fileset(fileset, dest)

        files.list_files.assert_called_once()
        fs.get.assert_called_once()
        get_call = fs.get.call_args
        assert get_call.args[0] == "default/qwen"
        assert get_call.args[1] == str(dest)
        assert get_call.kwargs["recursive"] is True
        assert dest.exists()

    def test_empty_fileset_returns_zero_stats_without_downloading(self, tmp_path: Path) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef

        files = _make_files_client()
        files.list_files.return_value.data.return_value = types.SimpleNamespace(data=[])
        runner = _make_runner(files)
        dest = tmp_path / "downloads"
        fileset = FileSetRef(workspace="default", name="empty")

        stats = runner.download_fileset(fileset, dest)

        assert stats.files_downloaded == 0
        assert stats.total_bytes == 0


class TestRun:
    def test_builds_sync_files_task_client_for_filesystem_transfers(self) -> None:
        import importlib

        file_io_run = importlib.import_module("nmp.customization_common.tasks.file_io.run")

        sync_client = MagicMock()
        sync_client.base_url = "http://nemo-platform.local"
        files = MagicMock()
        jobs = MagicMock()
        progress_reporter = MagicMock()
        runner = MagicMock()
        config = types.SimpleNamespace(
            upload=[],
            download=[],
            model_dump_json=lambda indent: "{}",
        )
        job_ctx = _make_job_ctx()

        with (
            patch.object(file_io_run, "get_config", return_value=config),
            patch.object(file_io_run, "get_task_nemo_client", return_value=sync_client) as get_task_nemo_client,
            patch.object(file_io_run.FilesClient, "from_client", return_value=files) as files_from_client,
            patch.object(file_io_run.JobsClient, "from_client", return_value=jobs) as jobs_from_client,
            patch.object(
                file_io_run.JobsServiceProgressReporter,
                "create_progress_reporter",
                return_value=progress_reporter,
            ) as create_progress_reporter,
            patch.object(file_io_run, "FileIORunner", return_value=runner) as runner_cls,
        ):
            result = file_io_run.run(
                job_ctx=job_ctx,
                service_source="rl",
                service_name="rl",
            )

        assert result == 0
        get_task_nemo_client.assert_called_once_with("rl")
        files_from_client.assert_called_once_with(sync_client)
        jobs_from_client.assert_called_once_with(sync_client)
        create_progress_reporter.assert_called_once_with(jobs, job_ctx)
        runner_cls.assert_called_once_with(
            files=files,
            progress_reporter=progress_reporter,
            job_ctx=job_ctx,
            service_source="rl",
        )
        runner.run_upload.assert_called_once_with([])
        runner.run_download.assert_called_once_with([])
        sync_client.close.assert_called_once()

    def test_owned_sync_client_closes(self) -> None:
        import importlib

        file_io_run = importlib.import_module("nmp.customization_common.tasks.file_io.run")

        sync_client = MagicMock()
        sync_client.base_url = "http://nemo-platform.local"
        config = types.SimpleNamespace(
            upload=[],
            download=[],
            model_dump_json=lambda indent: "{}",
        )

        with (
            patch.object(file_io_run, "get_config", return_value=config),
            patch.object(file_io_run, "get_task_nemo_client", return_value=sync_client),
            patch.object(file_io_run.FilesClient, "from_client", return_value=MagicMock()),
            patch.object(file_io_run.JobsClient, "from_client", return_value=MagicMock()),
            patch.object(file_io_run.JobsServiceProgressReporter, "create_progress_reporter", return_value=MagicMock()),
            patch.object(file_io_run, "FileIORunner", return_value=MagicMock()),
        ):
            result = file_io_run.run(
                job_ctx=_make_job_ctx(),
                service_source="rl",
                service_name="rl",
            )

        assert result == 0
        sync_client.close.assert_called_once()


@pytest.fixture
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip tenacity's exponential backoff so retry tests don't sleep for seconds.

    ``tenacity.nap.sleep`` resolves ``time.sleep`` per call, so patching it here
    takes effect even though the retry policy was bound at decoration time.
    """
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _seconds: None)


class TestCreateFilesetRetry:
    """Create goes through the typed client, so it must catch the typed client's errors."""

    def test_retries_transport_error_wrapped_by_the_client(self, no_retry_backoff) -> None:
        from nemo_platform_plugin.client.errors import NemoTransportError
        from nmp.customization_common.schemas.file_io import FileSetRef

        files = _make_files_client()
        files.create_fileset.side_effect = [NemoTransportError(httpx.ConnectError("refused")), MagicMock()]
        runner = _make_runner(files)

        runner.create_fileset(FileSetRef(workspace="default", name="models"))

        assert files.create_fileset.call_count == 2

    def test_retries_rate_limit_wrapped_by_the_client(self, no_retry_backoff) -> None:
        from nemo_platform_plugin.client.errors import RateLimitError
        from nmp.customization_common.schemas.file_io import FileSetRef

        response = httpx.Response(429, request=httpx.Request("POST", "http://test/filesets"), json={"detail": "slow"})
        files = _make_files_client()
        files.create_fileset.side_effect = [RateLimitError(response), MagicMock()]
        runner = _make_runner(files)

        runner.create_fileset(FileSetRef(workspace="default", name="models"))

        assert files.create_fileset.call_count == 2


class TestUploadRetry:
    """The task layer owns upload retries.

    The typed client cannot retry a streaming upload — the body is a one-shot
    iterator, and replaying it under the original Content-Length makes h11 abort
    the request. So the client raises through, and this layer, which rebuilds the
    request from the source file on every attempt, is where the retry belongs.
    """

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_retries_transport_error_wrapped_by_the_client(self, mock_fs_cls, tmp_path: Path, no_retry_backoff) -> None:
        from nemo_platform_plugin.client.errors import NemoTransportError
        from nmp.customization_common.schemas.file_io import FileSetRef

        src = _make_dir(tmp_path)
        fs = MagicMock()
        fs.put.side_effect = [NemoTransportError(httpx.ReadTimeout("timed out")), None]
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())

        runner.upload_fileset(FileSetRef(workspace="default", name="models"), src.resolve())

        assert fs.put.call_count == 2

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_retries_server_error_wrapped_by_the_client(self, mock_fs_cls, tmp_path: Path, no_retry_backoff) -> None:
        from nemo_platform_plugin.client.errors import InternalServerError
        from nmp.customization_common.schemas.file_io import FileSetRef

        src = _make_dir(tmp_path)
        response = httpx.Response(503, request=httpx.Request("PUT", "http://test/upload"), json={"detail": "down"})
        fs = MagicMock()
        fs.put.side_effect = [InternalServerError(response), None]
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())

        runner.upload_fileset(FileSetRef(workspace="default", name="models"), src.resolve())

        assert fs.put.call_count == 2

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_retries_rate_limit_wrapped_by_the_client(self, mock_fs_cls, tmp_path: Path, no_retry_backoff) -> None:
        """429 is in the client's retryable statuses, but it cannot act on it here."""
        from nemo_platform_plugin.client.errors import RateLimitError
        from nmp.customization_common.schemas.file_io import FileSetRef

        src = _make_dir(tmp_path)
        response = httpx.Response(429, request=httpx.Request("PUT", "http://test/upload"), json={"detail": "slow down"})
        fs = MagicMock()
        fs.put.side_effect = [RateLimitError(response), None]
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())

        runner.upload_fileset(FileSetRef(workspace="default", name="models"), src.resolve())

        assert fs.put.call_count == 2

    @patch("nmp.customization_common.tasks.file_io.run.FilesetFileSystem")
    def test_gives_up_as_a_file_upload_error(self, mock_fs_cls, tmp_path: Path, no_retry_backoff) -> None:
        from nemo_platform_plugin.client.errors import NemoTransportError
        from nmp.customization_common.schemas.file_io import FileSetRef, FileUploadError

        src = _make_dir(tmp_path)
        fs = MagicMock()
        fs.put.side_effect = NemoTransportError(httpx.ReadTimeout("timed out"))
        mock_fs_cls.return_value = fs
        runner = _make_runner(_make_files_client())

        with pytest.raises(FileUploadError):
            runner.upload_fileset(FileSetRef(workspace="default", name="models"), src.resolve())

        assert fs.put.call_count == 3
