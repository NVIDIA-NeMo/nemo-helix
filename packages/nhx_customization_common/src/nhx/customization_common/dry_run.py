# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of a customization dry run (create with ``?dry_run=true``).

A dry run checks that every step's execution profile is configured, that a GPU step's
profile has GPUs, and that each step's image exists, through the jobs API, which asks the
profile's backend. It reads the first rows of each file training would read and checks them
with the backend's own validator, reading byte ranges from the Files service, so nothing is
downloaded to job storage and nothing is created.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import ExecutionProfileGPUs, JobDryRunCheck
from nhx.customization_common.schemas.file_io import FileSetRef

ROWS_TO_CHECK = 10
_GPU_PROVIDERS = ("gpu", "gpu_distributed")
_CHUNK_BYTES = 64 * 1024
# A file whose first rows are longer than this is left unchecked rather than read in full.
_MAX_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class DatasetRows:
    """A dataset a job reads, and how training picks and checks its files."""

    #: What the check reports, e.g. ``"training dataset"``.
    name: str
    #: The fileset reference from the job spec.
    fileset: str
    #: Training's own file picker: relative paths in, the files training reads out.
    select_files: Callable[[list[str]], list[str]]
    #: Training's own row validator: ``(lines, source)``; raises on a bad row.
    validate_lines: Callable[[list[str], str], object]


async def check_execution_profiles(platform_spec: HelixJobSpec, async_sdk: AsyncNemoClient) -> JobDryRunCheck:
    """Check that every step's provider and profile is configured on the platform."""
    profiles = await AsyncJobsClient.from_client(async_sdk).list_execution_profiles()
    available = {(profile.provider, profile.profile) for profile in profiles}
    used = _step_profiles(platform_spec)
    missing = sorted(f"{provider}/{profile}" for provider, profile in used - available)
    if missing:
        configured = ", ".join(sorted(f"{provider}/{profile}" for provider, profile in available)) or "none"
        return JobDryRunCheck(
            name="execution profile",
            passed=False,
            message=f"Not configured on this platform: {', '.join(missing)}. Configured: {configured}.",
        )
    configured_used = ", ".join(sorted(f"{provider}/{profile}" for provider, profile in used))
    return JobDryRunCheck(name="execution profile", passed=True, message=f"Configured: {configured_used}.")


async def check_gpus(platform_spec: HelixJobSpec, async_sdk: AsyncNemoClient) -> JobDryRunCheck | None:
    """Check that the profile of every GPU step has GPUs; None when no step needs one.

    A profile whose GPUs cannot be read passes as not checked: the platform's not knowing
    is no reason to refuse the job.
    """
    profiles = sorted(
        (provider, profile) for provider, profile in _step_profiles(platform_spec) if provider in _GPU_PROVIDERS
    )
    if not profiles:
        return None
    jobs = AsyncJobsClient.from_client(async_sdk)
    results = await asyncio.gather(*(_profile_gpus(jobs, provider, profile) for provider, profile in profiles))
    failed = [message for passed, message in results if not passed]
    return JobDryRunCheck(name="GPUs", passed=not failed, message=" ".join(failed or [m for _, m in results]))


async def check_images(platform_spec: HelixJobSpec, async_sdk: AsyncNemoClient) -> JobDryRunCheck | None:
    """Check that each step's image exists for its profile, without pulling it; None when no step names one.

    An image that cannot be checked passes as not checked.
    """
    images = sorted(
        {
            (step.executor.provider, step.executor.profile, image)
            for step in platform_spec.steps
            if (image := getattr(getattr(step.executor, "container", None), "image", None))
        }
    )
    if not images:
        return None
    jobs = AsyncJobsClient.from_client(async_sdk)
    results = await asyncio.gather(*(_profile_image(jobs, *image) for image in images))
    failed = [message for passed, message in results if not passed]
    return JobDryRunCheck(name="images", passed=not failed, message=" ".join(failed or [m for _, m in results]))


def _step_profiles(platform_spec: HelixJobSpec) -> set[tuple[str, str]]:
    return {(step.executor.provider, step.executor.profile) for step in platform_spec.steps}


async def _profile_gpus(jobs: AsyncJobsClient, provider: str, profile: str) -> tuple[bool, str]:
    try:
        [result] = (
            await jobs.get_execution_profile_gpus(query_params={"provider": provider, "profile": profile})
        ).data()
    except (NemoClientError, ValueError) as exc:
        return True, f"{provider}/{profile}: GPUs not checked ({exc})."
    return result.status != "none", describe_profile_gpus(result)


def describe_profile_gpus(result: ExecutionProfileGPUs) -> str:
    """One line on a profile's GPUs: what it has, that it has none, or why they were not read."""
    where = f"{result.provider}/{result.profile}"
    if result.status == "available":
        found = ", ".join(
            f"{gpu.count}x {gpu.name}" + (f" (compute {gpu.compute_capability})" if gpu.compute_capability else "")
            for gpu in result.gpus
        )
        return f"{where}: {found}."
    if result.status == "none":
        return f"{where} has no GPU a job can use: {result.message}"
    return f"{where}: GPUs not checked ({result.message})."


async def _profile_image(jobs: AsyncJobsClient, provider: str, profile: str, image: str) -> tuple[bool, str]:
    try:
        result = (
            await jobs.check_execution_profile_image(
                query_params={"provider": provider, "profile": profile, "image": image}
            )
        ).data()
    except NemoClientError as exc:
        return True, f"{image}: not checked ({exc})."
    if result.status == "exists":
        return True, f"{image}: {result.message}"
    if result.status == "missing":
        return False, f"{image} is missing for {provider}/{profile}: {result.message}"
    return True, f"{image}: not checked ({result.message})."


async def check_dataset_rows(
    datasets: list[DatasetRows], workspace: str, async_sdk: AsyncNemoClient
) -> list[JobDryRunCheck]:
    """Check the first rows of every file each dataset's training would read."""
    files = AsyncFilesClient.from_client(async_sdk)
    return [await _check_fileset(files, dataset, workspace) for dataset in datasets]


def check_local_dataset_rows(dataset: DatasetRows, path: Path) -> JobDryRunCheck:
    """Check a local file or directory the way :func:`check_dataset_rows` checks a fileset."""
    try:
        if path.is_file():
            selected = {path.name: path}
        else:
            listing = {f.relative_to(path).as_posix(): f for f in path.rglob("*") if f.is_file()}
            selected = {rel: listing[rel] for rel in dataset.select_files(sorted(listing))}
        for rel, file in selected.items():
            with file.open("rb") as handle:
                head = handle.read(_MAX_BYTES)
            _validate_head(dataset, head, at_end=len(head) == file.stat().st_size, source=str(path / rel))
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, list(selected))


async def _check_fileset(files: AsyncFilesClient, dataset: DatasetRows, workspace: str) -> JobDryRunCheck:
    try:
        ref = FileSetRef.model_validate(dataset.fileset)
        fileset_workspace = ref.workspace or workspace
        prefix = ref.path or ""
        listing = (await files.list_files(workspace=fileset_workspace, name=ref.name)).data().data
        # Paths relative to the reference, as the download step lays them out.
        sizes = {
            file.path.lstrip("/").removeprefix(prefix): file.size
            for file in listing
            if file.path.lstrip("/").startswith(prefix)
        }
        selected = dataset.select_files(sorted(sizes))
        for rel in selected:
            path = f"{prefix}{rel}"
            head = await _read_head(files, fileset_workspace, ref.name, path, sizes[rel])
            source = f"{fileset_workspace}/{ref.name}#{path}"
            _validate_head(dataset, head, at_end=len(head) >= sizes[rel], source=source)
    except Exception as e:
        return JobDryRunCheck(name=dataset.name, passed=False, message=str(e))
    return _passed(dataset, selected)


async def _read_head(files: AsyncFilesClient, workspace: str, name: str, path: str, size: int) -> bytes:
    """Read from the start of a file until it holds :data:`ROWS_TO_CHECK` complete lines."""
    data = b""
    while len(data) < min(size, _MAX_BYTES) and data.count(b"\n") < ROWS_TO_CHECK:
        end = min(len(data) + _CHUNK_BYTES, size)
        response = await files.with_headers({"Range": f"bytes={len(data)}-{end - 1}"}).download_file(
            workspace=workspace, name=name, path=path
        )
        data += await response.read()
    return data


def _validate_head(dataset: DatasetRows, head: bytes, *, at_end: bool, source: str) -> None:
    """Validate the first :data:`ROWS_TO_CHECK` complete rows in *head*."""
    if at_end and not head.strip():
        raise ValueError(f"{source} is empty.")
    lines = head.split(b"\n")
    if not at_end:
        lines = lines[:-1]  # The last line may be cut off mid-row.
    try:
        text = [line.decode("utf-8") for line in lines]
    except UnicodeDecodeError as e:
        raise ValueError(f"{source} is not UTF-8 text: {e}") from e
    rows, kept = 0, []
    for line in text:
        if rows == ROWS_TO_CHECK:
            break
        kept.append(line)
        rows += bool(line.strip())
    if rows == 0:
        return  # The first row is longer than _MAX_BYTES; training validates it.
    dataset.validate_lines(kept, source)


def _passed(dataset: DatasetRows, files: list[str]) -> JobDryRunCheck:
    return JobDryRunCheck(
        name=dataset.name,
        passed=True,
        message=f"The first {ROWS_TO_CHECK} rows of {', '.join(files)} match what training reads.",
    )
