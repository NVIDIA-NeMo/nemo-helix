# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File I/O task entry point.

Handles file operations between NeMo Platform Files Service and the job's shared PVC.

Each backend's compiler sets ``--service-source`` and ``--service-name`` via
``FILE_IO_TASK_COMMAND`` in that backend's ``images.py`` (the job step passes
these flags; do not hardcode one backend when documenting or invoking locally):

- automodel: ``--service-source automodel --service-name customizer``
- unsloth: ``--service-source unsloth --service-name unsloth``
- rl: ``--service-source rl --service-name rl``

Usage (example — match the backend you are exercising)::

    export NEMO_JOB_STEP_CONFIG_FILE_PATH=<path to job_step_config.json>
    python -m nmp.customization_common.tasks.file_io --service-source automodel --service-name customizer
"""

import logging
from pathlib import Path

import httpx
from filesets.filesystem.filesystem import FilesetFileSystem, build_fileset_ref
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.errors import (
    ConflictError,
    NemoTransportError,
    NotFoundError,
    RateLimitError,
)
from nemo_platform_plugin.client.errors import (
    InternalServerError as ClientInternalServerError,
)
from nemo_platform_plugin.client.types import RetryPolicy
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.metadata import FilesetMetadata
from nemo_platform_plugin.files.types import CreateFilesetRequest, FilesetFileOutput, UpdateFilesetRequest
from nemo_platform_plugin.jobs.client import JobsClient
from nmp.common.client_factory import get_task_nemo_client
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.customization_common.schemas.file_io import (
    DownloadItem,
    DownloadStats,
    FileDownloadError,
    FileSetRef,
    FileUploadError,
    PathTraversalError,
    TaskPhase,
    UploadItem,
    UploadStats,
)
from nmp.customization_common.service.context import NMPJobContext
from nmp.customization_common.tasks.file_io.callbacks import (
    CompositeCallback,
    FileDownloadProgressCallback,
    FileUploadProgressCallback,
    TqdmPerFileDownloadCallback,
    TqdmPerFileUploadCallback,
)
from nmp.customization_common.tasks.file_io_progress_reporter import JobsServiceProgressReporter, ProgressReporter
from nmp.customization_common.tasks.file_io_utils import (
    filesystem_sdk_error_handler,
    get_config,
    sdk_error_handler,
    validate_safe_path,
    validate_storage_path,
)
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CREATE_FILESET_TIMEOUT = 10.0
LIST_FILES_TIMEOUT = httpx.Timeout(10.0, connect=10.0)

DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=5 * 60)
UPLOAD_TIMEOUT = httpx.Timeout(30.0, write=10 * 60, read=5 * 60)

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0

# Transient failures worth another attempt, for every Files operation this task
# runs. Uploads stream a one-shot body, so once it is on the wire the typed client
# can no longer retry them itself — this is the layer that rebuilds the request
# from the source file, so it has to catch what the client raises. Those are the
# Nemo* entries: they wrap transport failures, 5xx responses (raise_for_status
# maps every 5xx to InternalServerError) and 429 respectively, and are not httpx
# types. Anything reaching this task through the typed client arrives as one of
# them, never as the bare httpx error underneath.
TRANSIENT_FILESYSTEM_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    NemoTransportError,
    ClientInternalServerError,
    RateLimitError,
)


class FileIORunner:
    """Runner for file I/O operations against the Files service."""

    def __init__(
        self,
        files: FilesClient,
        progress_reporter: ProgressReporter,
        job_ctx: NMPJobContext,
        *,
        service_source: str,
    ):
        self.files = files
        self.progress_reporter = progress_reporter
        self.job_ctx = job_ctx
        self.service_source = service_source

    def _workspace_for(self, fileset: FileSetRef) -> str:
        return fileset.workspace or self.job_ctx.workspace

    def list_fileset_files(self, fileset: FileSetRef) -> list[FilesetFileOutput]:
        """List files in a FileSet. Returns a list of ``FilesetFileOutput`` objects."""
        try:
            with sdk_error_handler(FileDownloadError, f"list files in fileset {fileset}", passthrough=(NotFoundError,)):
                response = (
                    self.files.with_options(timeout=LIST_FILES_TIMEOUT)
                    .list_files(
                        name=fileset.name,
                        workspace=self._workspace_for(fileset),
                    )
                    .data()
                )
                logger.info(f"Found {len(response.data)} files in FileSet {fileset!s}")
                return response.data
        except NotFoundError as e:
            raise FileDownloadError(
                f"FileSet {fileset!s} not found. Please ensure the FileSet exists and contains the expected files.",
            ) from e

    def download_fileset(self, fileset: FileSetRef, dest_dir: Path) -> DownloadStats:
        """Download all files from a FileSet to a destination directory."""
        fileset_name = str(fileset)

        files = self.list_fileset_files(fileset)

        if not files:
            logger.warning(f"FileSet {fileset_name} contains no files")
            return DownloadStats()

        total_files = len(files)
        total_size = sum(f.size for f in files)

        dest_dir.mkdir(parents=True, exist_ok=True)

        file_sizes = {f.path.lstrip("/"): f.size for f in files}

        with filesystem_sdk_error_handler(
            FileDownloadError,
            f"download from '{fileset_name}' to '{dest_dir}'",
        ):
            stats = self._download_with_retry(
                fileset_name=fileset.name,
                fileset_workspace=self._workspace_for(fileset),
                dest_dir=str(dest_dir),
                fileset_display_name=fileset_name,
                dest_path=dest_dir,
                file_sizes=file_sizes,
                total_files=total_files,
                total_size=total_size,
            )

        logger.info(f"Download complete: {stats.files_downloaded} files, {stats.total_bytes} bytes")
        return stats

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=INITIAL_BACKOFF_SECONDS, max=MAX_BACKOFF_SECONDS),
        retry=retry_if_exception_type(TRANSIENT_FILESYSTEM_EXCEPTIONS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _download_with_retry(
        self,
        fileset_name: str,
        fileset_workspace: str,
        dest_dir: str,
        fileset_display_name: str,
        dest_path: Path,
        file_sizes: dict[str, int],
        total_files: int,
        total_size: int,
    ) -> DownloadStats:
        """Internal method with retry logic for downloading from FilesetFileSystem."""
        stats = DownloadStats()
        tqdm_callback = TqdmPerFileDownloadCallback(
            dest_path=dest_path,
            fileset_path=fileset_display_name,
            file_sizes=file_sizes,
        )
        jobs_callback = FileDownloadProgressCallback(
            progress_reporter=self.progress_reporter,
            fileset_name=fileset_display_name,
            total_files=total_files,
            total_size=total_size,
            stats=stats,
        )
        composite_callback = CompositeCallback(tqdm_callback, jobs_callback)

        FilesetFileSystem(client=self.files.with_options(timeout=DOWNLOAD_TIMEOUT)).get(
            build_fileset_ref("", workspace=fileset_workspace, fileset=fileset_name),
            dest_dir,
            recursive=True,
            callback=composite_callback,
        )
        return stats

    def upload_fileset(self, fileset: FileSetRef, src_path: Path) -> UploadStats:
        """Upload all files from a source path (file or directory) to a FileSet."""
        fileset_name = str(fileset)

        if src_path.is_dir():
            local_path = f"{src_path}/"
            remote_path = ""
        else:
            local_path = str(src_path)
            remote_path = src_path.name

        with filesystem_sdk_error_handler(
            FileUploadError,
            f"upload from '{src_path}' to '{fileset_name}'",
        ):
            stats = self._upload_with_retry(
                local_path=local_path,
                remote_path=remote_path,
                fileset_name=fileset.name,
                fileset_workspace=self._workspace_for(fileset),
                fileset_display_name=fileset_name,
                src_path=src_path,
            )

        logger.info(f"Upload complete: {stats.files_uploaded} files, {stats.total_bytes} bytes")
        return stats

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=INITIAL_BACKOFF_SECONDS, max=MAX_BACKOFF_SECONDS),
        retry=retry_if_exception_type(TRANSIENT_FILESYSTEM_EXCEPTIONS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
    )
    def _upload_with_retry(
        self,
        local_path: str,
        remote_path: str,
        fileset_name: str,
        fileset_workspace: str,
        fileset_display_name: str,
        src_path: Path,
    ) -> UploadStats:
        """Internal method with retry logic for uploading to FilesetFileSystem."""
        stats = UploadStats()
        tqdm_callback = TqdmPerFileUploadCallback(src_path=src_path)
        jobs_callback = FileUploadProgressCallback(
            progress_reporter=self.progress_reporter,
            src_path=src_path,
            fileset_name=fileset_display_name,
            stats=stats,
        )
        composite_callback = CompositeCallback(tqdm_callback, jobs_callback)

        FilesetFileSystem(client=self.files.with_options(timeout=UPLOAD_TIMEOUT)).put(
            local_path,
            build_fileset_ref(remote_path, workspace=fileset_workspace, fileset=fileset_name),
            recursive=True,
            callback=composite_callback,
        )
        return stats

    def create_fileset(self, fileset: FileSetRef, metadata: dict | None = None) -> None:
        """Create a FileSet. Skip if it already exists."""
        with sdk_error_handler(FileUploadError, f"create fileset {fileset}", passthrough=(ConflictError,)):
            self._create_fileset_with_retry(fileset, metadata)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=INITIAL_BACKOFF_SECONDS, max=MAX_BACKOFF_SECONDS),
        retry=retry_if_exception_type(TRANSIENT_FILESYSTEM_EXCEPTIONS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _create_fileset_with_retry(self, fileset: FileSetRef, metadata: dict | None = None) -> None:
        """Internal method with retry logic for creating a FileSet."""
        files = self.files.with_options(timeout=CREATE_FILESET_TIMEOUT, retry=RetryPolicy(max_retries=0))
        workspace = self._workspace_for(fileset)
        try:
            body_kwargs: dict = {
                "name": fileset.name,
                "custom_fields": {"service_source": self.service_source},
            }
            if metadata is not None:
                body_kwargs["metadata"] = FilesetMetadata.model_validate(metadata)
            result = files.create_fileset(workspace=workspace, body=CreateFilesetRequest(**body_kwargs)).data()
            logger.info(f"Created FileSet: {result.workspace}/{result.name}")
        except ConflictError:
            if metadata is not None:
                try:
                    files.update_fileset(
                        workspace=workspace,
                        name=fileset.name,
                        body=UpdateFilesetRequest(metadata=FilesetMetadata.model_validate(metadata)),
                    )
                    logger.info(f"Patched existing FileSet metadata: {workspace}/{fileset.name}")
                except Exception as e:
                    logger.warning(
                        f"Could not patch metadata on existing fileset {workspace}/{fileset.name}: {e}. "
                        "Upload will continue; downstream consumers may lack the latest metadata.",
                    )

    def run_download(self, downloads: list[DownloadItem]) -> None:
        """Execute download operations."""
        if not downloads:
            logger.info("No downloads configured, skipping download operation")
            return

        storage_path = validate_storage_path(self.job_ctx.storage_path)

        logger.info(f"Starting download operation: {len(downloads)} fileset(s) to download")

        self.progress_reporter.update_progress(
            status=PlatformJobStatus.ACTIVE,
            status_details={
                "phase": TaskPhase.DOWNLOADING,
                "total_filesets": len(downloads),
                "completed_filesets": 0,
            },
        )

        total_stats = DownloadStats()

        for idx, item in enumerate(downloads):
            fileset = item.src
            dest_dir = validate_safe_path(storage_path, item.dest)

            logger.info(f"[{idx + 1}/{len(downloads)}] Downloading from {fileset!s} to {dest_dir}")

            self.progress_reporter.update_progress(
                status=PlatformJobStatus.ACTIVE,
                status_details={
                    "phase": TaskPhase.DOWNLOADING,
                    "total_filesets": len(downloads),
                    "completed_filesets": idx,
                    "current_fileset": f"{fileset!s}",
                },
            )

            stats = self.download_fileset(fileset, dest_dir)
            total_stats.files_downloaded += stats.files_downloaded
            total_stats.total_bytes += stats.total_bytes

            logger.info(f"FileSet download complete: {stats.files_downloaded} files, {stats.total_bytes} bytes")

        logger.info(
            f"All downloads complete: {total_stats.files_downloaded} files, {total_stats.total_bytes} bytes total",
        )

    def run_upload(self, uploads: list[UploadItem]) -> None:
        """Execute upload operations."""
        if not uploads:
            logger.info("No uploads configured, skipping upload operation")
            return

        storage_path = validate_storage_path(self.job_ctx.storage_path)

        logger.info(f"Starting upload operation: {len(uploads)} fileset(s) to upload")

        self.progress_reporter.update_progress(
            status=PlatformJobStatus.ACTIVE,
            status_details={
                "phase": TaskPhase.UPLOADING,
                "total_filesets": len(uploads),
                "completed_filesets": 0,
            },
        )

        total_stats = UploadStats()

        for idx, item in enumerate(uploads):
            if item.dest.workspace is None:
                item.dest.workspace = self.job_ctx.workspace
            fileset = item.dest
            src_path = validate_safe_path(storage_path, item.src)
            if not src_path.exists():
                raise FileUploadError(f"Source path does not exist: {src_path}. Ensure the source path exists.")
            if not src_path.is_dir() and not src_path.is_file():
                raise FileUploadError(
                    f"Source path is not a file or directory: {src_path}. "
                    "Ensure the source path is a file or directory.",
                )

            logger.info(f"[{idx + 1}/{len(uploads)}] Uploading from {src_path} to {fileset!s}")

            self.progress_reporter.update_progress(
                status=PlatformJobStatus.ACTIVE,
                status_details={
                    "phase": TaskPhase.UPLOADING,
                    "total_filesets": len(uploads),
                    "completed_filesets": idx,
                    "current_fileset": str(fileset),
                },
            )

            self.create_fileset(fileset, metadata=item.metadata)

            stats = self.upload_fileset(fileset, src_path)
            total_stats.files_uploaded += stats.files_uploaded
            total_stats.total_bytes += stats.total_bytes

            logger.info(f"FileSet upload complete: {stats.files_uploaded} files, {stats.total_bytes} bytes")

        logger.info(f"All uploads complete: {total_stats.files_uploaded} files, {total_stats.total_bytes} bytes total")


def run(
    client: NemoClient | None = None,
    job_ctx: NMPJobContext | None = None,
    *,
    service_source: str,
    service_name: str,
) -> int:
    """Execute the file I/O task."""
    job_ctx = job_ctx or NMPJobContext.from_env()
    validate_storage_path(job_ctx.storage_path)

    client_owned = client is None
    progress_reporter: ProgressReporter | None = None
    try:
        client = client or get_task_nemo_client(service_name)
        files = FilesClient.from_client(client)
        jobs = JobsClient.from_client(client)
        progress_reporter = JobsServiceProgressReporter.create_progress_reporter(jobs, job_ctx)
        runner = FileIORunner(
            files=files,
            progress_reporter=progress_reporter,
            job_ctx=job_ctx,
            service_source=service_source,
        )

        config = get_config(job_ctx.config_path)

        logger.info(f"Starting file I/O task with job context: {job_ctx}")
        logger.info(f"Config: {config.model_dump_json(indent=2)}")
        logger.info(f"NeMo Platform service URL: {client.base_url}")

        runner.run_upload(config.upload)
        runner.run_download(config.download)

        progress_reporter.update_progress(
            status=PlatformJobStatus.COMPLETED,
            status_details={"phase": TaskPhase.COMPLETED, "message": "File I/O task completed successfully"},
        )

        return 0
    except PathTraversalError as e:
        logger.error(f"Security error - path traversal detected: {e}")
        if progress_reporter:
            progress_reporter.update_progress(
                status=PlatformJobStatus.ERROR,
                error_details={"message": str(e), "type": type(e).__name__},
            )
        return 1
    except (FileDownloadError, FileUploadError) as e:
        logger.exception(f"File operation failed: {e}")
        if progress_reporter:
            progress_reporter.update_progress(
                status=PlatformJobStatus.ERROR,
                error_details={"message": str(e), "type": type(e).__name__},
            )
        return 1
    except Exception as e:
        logger.exception(f"File I/O task failed: {e}")
        if progress_reporter:
            progress_reporter.update_progress(
                status=PlatformJobStatus.ERROR,
                error_details={"message": str(e), "type": type(e).__name__},
            )
        return 1
    finally:
        if client_owned and client is not None:
            try:
                client.close()
            except Exception:
                logger.warning("Failed to close sync platform client during file I/O task cleanup", exc_info=True)
