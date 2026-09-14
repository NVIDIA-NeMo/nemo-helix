# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from filesets import AsyncFilesetFileSystem, FilesetFileSystem, parse_fileset_ref
from nemo_platform_plugin.client.errors import ConflictError as ClientConflictError
from nemo_platform_plugin.client.errors import NemoClientError
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_platform_plugin.jobs.file_manager import AsyncFilesetFileManager, FilesetFileManager, TmpDirPath
from nemo_platform_plugin.jobs.schemas import PlatformJobResultCreateRequest, PlatformJobResultResponse
from nemo_platform_plugin.jobs.types import job_artifact_base_path

logger = logging.getLogger(__name__)


class CreateJobResultError(Exception): ...


class FileDoesNotExist(Exception): ...


class BaseResultManager:
    """
    Base class for sync and async versions of the ResultManager. If there is any common code that can be used across
    both versions of the ResultManager, try to lift it up into this class to reduce duplication.
    """

    job_name: str
    workspace: str
    attempt_id: str | None

    def _validate_local_path(self, artifact_local_path: str | Path) -> Path:
        if isinstance(artifact_local_path, str):
            artifact_local_path = Path(artifact_local_path)

        if not artifact_local_path.exists():
            raise FileDoesNotExist(f"No file exists at: {artifact_local_path}")

        return artifact_local_path

    def _result_remote_path(self, attempt_id: str, result_name: str, base: str | None = None) -> str:
        """Build the remote path for a result artifact, optionally nested under a per-job base folder."""
        prefix = f"{base}/" if base else ""
        return f"{prefix}results/{attempt_id}/{result_name}"


@dataclass
class ResultManager(BaseResultManager):
    job_name: str
    workspace: str
    files_client: FilesClient
    jobs_client: JobsClient
    attempt_id: str | None = field(default=None)

    def _fetch_job_metadata(self) -> tuple[str, str, str | None]:
        """Fetch job and return (attempt_id, fileset_name, output_location)."""
        job = self.jobs_client.get_job(name=self.job_name, workspace=self.workspace).data()
        attempt_id = self.attempt_id if self.attempt_id is not None else job.attempt_id
        return attempt_id, job.fileset, job.output_location

    def _create_file_manager(self, fileset_name: str) -> FilesetFileManager:
        """Create a file manager for the given fileset."""
        return FilesetFileManager(
            workspace=self.workspace,
            fileset_name=fileset_name,
            filesystem=FilesetFileSystem(client=self.files_client),
        )

    def create_result(
        self,
        result_name: str,
        artifact_local_path: str | Path,
        ignore_patterns: list[str] | str | None = None,
    ) -> PlatformJobResultResponse:
        attempt_id, fileset_name, output_location = self._fetch_job_metadata()
        file_manager = self._create_file_manager(fileset_name)
        file_manager.validate_storage()
        artifact_local_path = self._validate_local_path(artifact_local_path)
        remote_path = self._result_remote_path(
            attempt_id, result_name, base=job_artifact_base_path(self.job_name, output_location)
        )
        artifact_url = file_manager.upload(
            local_path=artifact_local_path, remote_path=remote_path, ignore_patterns=ignore_patterns
        )
        try:
            job_result = self.jobs_client.create_job_result(
                name=result_name,
                job=self.job_name,
                workspace=self.workspace,
                body=PlatformJobResultCreateRequest(
                    artifact_url=artifact_url,
                    artifact_storage_type=file_manager.storage_type(),
                ),
            ).data()
        except ClientConflictError:
            # Result already exists - fetch and return the existing one
            # This supports the use case of saving partial results across multiple batches
            job_result = self.jobs_client.get_job_result(
                name=result_name,
                job=self.job_name,
                workspace=self.workspace,
            ).data()
        except NemoClientError as e:
            msg = f"Error creating job result: {str(e)}"
            logger.exception(msg)
            raise CreateJobResultError(msg) from e

        return job_result

    def download_artifact(self, artifact_url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        _, fileset_name, _ = parse_fileset_ref(artifact_url, workspace_fallback=self.workspace)
        if fileset_name is None:
            raise ValueError(f"Could not extract fileset name from URL: {artifact_url}")
        file_manager = self._create_file_manager(fileset_name)
        return file_manager.download_from_url(url=artifact_url, local_dir=local_dir)


@dataclass
class AsyncResultManager(BaseResultManager):
    job_name: str
    workspace: str
    files_client: AsyncFilesClient
    jobs_client: AsyncJobsClient
    attempt_id: str | None = field(default=None)

    async def _fetch_job_metadata(self) -> tuple[str, str, str | None]:
        """Fetch job and return (attempt_id, fileset_name, output_location)."""
        job = (await self.jobs_client.get_job(name=self.job_name, workspace=self.workspace)).data()
        attempt_id = self.attempt_id if self.attempt_id is not None else job.attempt_id
        return attempt_id, job.fileset, job.output_location

    def _create_file_manager(self, fileset_name: str) -> AsyncFilesetFileManager:
        """Create a file manager for the given fileset."""
        return AsyncFilesetFileManager(
            workspace=self.workspace,
            fileset_name=fileset_name,
            filesystem=AsyncFilesetFileSystem(client=self.files_client),
        )

    async def create_result(
        self,
        result_name: str,
        artifact_local_path: str | Path,
        ignore_patterns: list[str] | str | None = None,
    ) -> PlatformJobResultResponse:
        attempt_id, fileset_name, output_location = await self._fetch_job_metadata()
        file_manager = self._create_file_manager(fileset_name)
        await file_manager.validate_storage()
        artifact_local_path = self._validate_local_path(artifact_local_path)
        remote_path = self._result_remote_path(
            attempt_id, result_name, base=job_artifact_base_path(self.job_name, output_location)
        )
        artifact_url = await file_manager.upload(
            local_path=artifact_local_path, remote_path=remote_path, ignore_patterns=ignore_patterns
        )
        try:
            job_result = (
                await self.jobs_client.create_job_result(
                    name=result_name,
                    job=self.job_name,
                    workspace=self.workspace,
                    body=PlatformJobResultCreateRequest(
                        artifact_url=artifact_url,
                        artifact_storage_type=file_manager.storage_type(),
                    ),
                )
            ).data()
        except ClientConflictError:
            # Result already exists - fetch and return the existing one
            # This supports the use case of saving partial results across multiple batches
            job_result = (
                await self.jobs_client.get_job_result(
                    name=result_name,
                    job=self.job_name,
                    workspace=self.workspace,
                )
            ).data()
        except NemoClientError as e:
            msg = f"Error creating job result: {str(e)}"
            logger.exception(msg)
            raise CreateJobResultError(msg) from e

        return job_result

    async def download_artifact(self, artifact_url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        _, fileset_name, _ = parse_fileset_ref(artifact_url, workspace_fallback=self.workspace)
        if fileset_name is None:
            raise ValueError(f"Could not extract fileset name from URL: {artifact_url}")
        file_manager = self._create_file_manager(fileset_name)
        return await file_manager.download_from_url(url=artifact_url, local_dir=local_dir)


def result_manager_factory(
    job_name: str,
    *,
    attempt_id: str | None = None,
    workspace: str | None = None,
    files_client: FilesClient,
    jobs_client: JobsClient | None = None,
) -> ResultManager:
    """Create a sync ResultManager for uploading job results."""
    resolved_workspace = files_client.resolve_workspace(workspace)
    return ResultManager(
        job_name=job_name,
        workspace=resolved_workspace,
        attempt_id=attempt_id,
        files_client=files_client,
        jobs_client=jobs_client or JobsClient.from_client(files_client),
    )


def async_result_manager_factory(
    job_name: str,
    *,
    attempt_id: str | None = None,
    workspace: str | None = None,
    files_client: AsyncFilesClient,
    jobs_client: AsyncJobsClient | None = None,
) -> AsyncResultManager:
    """Create a ResultManager for uploading job results.

    Args:
        job_name: Name of the job to create results for.
        attempt_id: Optional attempt ID for the job.
        workspace: Job workspace. If not provided, falls back to the client workspace resolution.
        files_client: Files service client.
        jobs_client: Jobs service client. If not provided, one is created from
            files_client's shared transport.
    """
    resolved_workspace = files_client.resolve_workspace(workspace)
    return AsyncResultManager(
        job_name=job_name,
        workspace=resolved_workspace,
        attempt_id=attempt_id,
        files_client=files_client,
        jobs_client=jobs_client or AsyncJobsClient.from_client(files_client),
    )


async def download_from_result_info(
    result_name: str,
    job_name: str,
    *,
    artifact_url: str,
    workspace: str | None = None,
    files_client: AsyncFilesClient,
) -> tuple[str, TmpDirPath]:
    """
    This is a helper composition function that creates a result_manager
    and prepares the artifact for download.

    Returns:
        A tuple containing:
          - filename (str): The final filename of the artifact (may include .tar.gz extension if compressed)
          - tmp_dir_path (TmpDirPath): Path object pointing to the downloaded artifact
    """
    result_manager = async_result_manager_factory(
        job_name=job_name,
        workspace=workspace,
        files_client=files_client,
    )

    tmp_dir_path = await result_manager.download_artifact(artifact_url=artifact_url)
    filename = result_name

    if tmp_dir_path.path.is_dir():
        filename = f"{filename}.tar.gz"
        tar_path = tmp_dir_path.tmp_dir / filename
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(tmp_dir_path.path, arcname=os.path.basename(tmp_dir_path.path))

        tmp_dir_path.path = tar_path

    return filename, tmp_dir_path
