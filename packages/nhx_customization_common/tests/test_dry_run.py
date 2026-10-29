# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import (
    ExecutionProfileGPU,
    ExecutionProfileGPUs,
    ExecutionProfileGPUStatus,
    ExecutionProfileImage,
    ExecutionProfileImageStatus,
    JobDryRunCheck,
)
from nhx.customization_common import dry_run
from nhx.customization_common.contributor import jobs as contributor_jobs
from nhx.customization_common.contributor.jobs import BaseSubmitJob
from nhx.customization_common.dry_run import (
    DatasetRows,
    check_dataset_rows,
    check_execution_profiles,
    check_gpus,
    check_images,
    check_local_dataset_rows,
)


class _FakeFiles:
    """Serves one fileset's files, honoring the Range header like the Files service."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.ranges: list[tuple[str, str]] = []
        self._range: str | None = None

    def with_headers(self, headers: dict[str, str]) -> "_FakeFiles":
        view = _FakeFiles(self.files)
        view.ranges = self.ranges
        view._range = headers["Range"]
        return view

    async def list_files(self, *, workspace: str, name: str) -> MagicMock:
        assert (workspace, name) == ("ws", "data")
        listing = [SimpleNamespace(path=path, size=len(content)) for path, content in self.files.items()]
        response = MagicMock()
        response.data.return_value = SimpleNamespace(data=listing)
        return response

    async def download_file(self, *, workspace: str, name: str, path: str) -> SimpleNamespace:
        assert self._range is not None, "a dry run must read byte ranges, not whole files"
        self.ranges.append((path, self._range))
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self._range)
        assert match is not None
        start, end = (int(n) for n in match.groups())

        async def read() -> bytes:
            return self.files[path][start : end + 1]

        return SimpleNamespace(read=read)


def _rows(n: int, row: dict | None = None) -> bytes:
    return b"".join(json.dumps(row or {"text": f"row {i}"}).encode() + b"\n" for i in range(n))


def _require_text(lines: list[str], source: str) -> int:
    for number, line in enumerate(lines, start=1):
        if line.strip() and "text" not in json.loads(line):
            raise ValueError(f"{source} line {number} has no 'text'")
    return len(lines)


def _dataset(fileset: str = "ws/data") -> DatasetRows:
    return DatasetRows(
        name="training dataset",
        fileset=fileset,
        select_files=lambda paths: [path for path in paths if path.startswith("train")],
        validate_lines=_require_text,
    )


async def _check(files: _FakeFiles, dataset: DatasetRows) -> dry_run.JobDryRunCheck:
    with patch.object(dry_run.AsyncFilesClient, "from_client", return_value=files):
        [check] = await check_dataset_rows([dataset], "ws", MagicMock())
    return check


async def test_checks_the_first_rows_of_the_files_training_reads() -> None:
    files = _FakeFiles({"train.jsonl": _rows(5_000), "notes.txt": b"not data"})
    check = await _check(files, _dataset())

    assert check.passed, check.message
    assert check.message == "The first 10 rows of train.jsonl match what training reads."
    # One 64 KB range covers 10 short rows; the rest of the file is never read.
    assert files.ranges == [("train.jsonl", "bytes=0-65535")]


async def test_a_bad_row_fails_with_its_line_number() -> None:
    content = _rows(2) + json.dumps({"prompt": "no text"}).encode() + b"\n" + _rows(20)
    check = await _check(_FakeFiles({"train.jsonl": content}), _dataset())

    assert not check.passed
    assert check.message == "ws/data#train.jsonl line 3 has no 'text'"


async def test_rows_past_the_first_ten_are_not_checked() -> None:
    content = _rows(10) + json.dumps({"bad": True}).encode() + b"\n"
    assert (await _check(_FakeFiles({"train.jsonl": content}), _dataset())).passed


async def test_reads_more_ranges_until_ten_rows_are_complete() -> None:
    long_row = {"text": "x" * 40_000}
    files = _FakeFiles({"train.jsonl": _rows(12, long_row)})
    assert (await _check(files, _dataset())).passed
    assert len(files.ranges) == 7  # ten 40 KB rows need just over 400 KB


async def test_a_fileset_path_scopes_the_files() -> None:
    files = _FakeFiles({"sft/train.jsonl": _rows(3), "train.jsonl": b"{broken"})
    check = await _check(files, _dataset("ws/data#sft"))

    assert check.passed, check.message
    assert [path for path, _ in files.ranges] == ["sft/train.jsonl"]


async def test_an_empty_file_or_no_training_files_fail() -> None:
    assert (await _check(_FakeFiles({"train.jsonl": b""}), _dataset())).message == "ws/data#train.jsonl is empty."

    def no_training(paths: list[str]) -> list[str]:
        raise ValueError("No training files found.")

    dataset = DatasetRows("training dataset", "ws/data", no_training, _require_text)
    check = await _check(_FakeFiles({"README.md": b"hi"}), dataset)
    assert (check.passed, check.message) == (False, "No training files found.")


def test_local_dataset_is_checked_the_same_way(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_bytes(_rows(2) + json.dumps({"x": 1}).encode() + b"\n")
    check = check_local_dataset_rows(_dataset(), tmp_path)
    assert (check.passed, check.message) == (False, f"{tmp_path}/train.jsonl line 3 has no 'text'")

    (tmp_path / "train.jsonl").write_bytes(_rows(2))
    assert check_local_dataset_rows(_dataset(), tmp_path / "train.jsonl").passed


def _platform_spec(*profiles: tuple[str, str]) -> HelixJobSpec:
    return HelixJobSpec.model_validate(
        {
            "steps": [
                {
                    "name": f"step-{i}",
                    "executor": {"provider": provider, "profile": profile, "container": {"image": "trainer:1"}},
                }
                for i, (provider, profile) in enumerate(profiles)
            ]
        }
    )


def _jobs(*profiles: tuple[str, str]) -> MagicMock:
    jobs = MagicMock()
    jobs.list_execution_profiles = AsyncMock(
        return_value=[SimpleNamespace(provider=provider, profile=profile) for provider, profile in profiles]
    )
    return jobs


async def test_execution_profiles_pass_when_every_step_profile_is_configured() -> None:
    with patch.object(
        dry_run.AsyncJobsClient, "from_client", return_value=_jobs(("gpu", "default"), ("cpu", "default"))
    ):
        check = await check_execution_profiles(_platform_spec(("gpu", "default"), ("cpu", "default")), MagicMock())

    assert (check.passed, check.message) == (True, "Configured: cpu/default, gpu/default.")


async def test_a_profile_the_platform_does_not_have_fails_and_lists_the_configured_ones() -> None:
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=_jobs(("cpu", "default"))):
        check = await check_execution_profiles(_platform_spec(("gpu", "default")), MagicMock())

    assert (check.passed, check.message) == (
        False,
        "Not configured on this platform: gpu/default. Configured: cpu/default.",
    )


def _steps(*steps: tuple[str, str, str]) -> HelixJobSpec:
    """A compiled job whose steps run (provider, profile, image)."""
    return HelixJobSpec.model_validate(
        {
            "steps": [
                {
                    "name": f"step-{i}",
                    "executor": {"provider": provider, "profile": profile, "container": {"image": image}},
                }
                for i, (provider, profile, image) in enumerate(steps)
            ]
        }
    )


def _gpus(status: ExecutionProfileGPUStatus, *gpus: ExecutionProfileGPU, message: str | None = None) -> MagicMock:
    result = ExecutionProfileGPUs(
        provider="gpu", profile="default", backend="docker", status=status, gpus=list(gpus), message=message
    )
    return MagicMock(data=MagicMock(return_value=[result]))


def _image(image: str, status: ExecutionProfileImageStatus, message: str) -> MagicMock:
    result = ExecutionProfileImage(provider="gpu", profile="default", image=image, status=status, message=message)
    return MagicMock(data=MagicMock(return_value=result))


_TRAINING = ("gpu", "default", "nvcr.io/nvidia/nemo/automodel:1")
_TASKS = ("cpu", "default", "nvcr.io/nvidia/nemo/tasks:1")
_H100 = ExecutionProfileGPU(name="NVIDIA H100 80GB HBM3", compute_capability="9.0", count=8)


async def test_gpus_are_read_only_for_gpu_steps_and_reported_by_model() -> None:
    jobs = MagicMock(get_execution_profile_gpus=AsyncMock(return_value=_gpus("available", _H100)))
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=jobs):
        check = await check_gpus(_steps(_TASKS, _TRAINING, _TASKS), MagicMock())

    assert check is not None
    assert (check.passed, check.message) == (True, "gpu/default: 8x NVIDIA H100 80GB HBM3 (compute 9.0).")
    jobs.get_execution_profile_gpus.assert_awaited_once_with(query_params={"provider": "gpu", "profile": "default"})


async def test_a_gpu_profile_without_gpus_fails() -> None:
    jobs = MagicMock(
        get_execution_profile_gpus=AsyncMock(return_value=_gpus("none", message="nvidia-smi listed no GPUs."))
    )
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=jobs):
        check = await check_gpus(_steps(_TRAINING), MagicMock())

    assert check is not None
    assert (check.passed, check.message) == (False, "gpu/default has no GPU a job can use: nvidia-smi listed no GPUs.")


async def test_gpus_that_cannot_be_read_pass_as_not_checked() -> None:
    jobs = MagicMock(get_execution_profile_gpus=AsyncMock(return_value=_gpus("unknown", message="RBAC missing.")))
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=jobs):
        check = await check_gpus(_steps(_TRAINING), MagicMock())

    assert check is not None
    assert (check.passed, check.message) == (True, "gpu/default: GPUs not checked (RBAC missing.).")


async def test_a_job_without_gpu_steps_has_no_gpu_check() -> None:
    assert await check_gpus(_steps(_TASKS), MagicMock()) is None


async def test_each_distinct_image_is_checked_and_a_missing_one_fails() -> None:
    def answer(*, query_params: dict[str, str]) -> MagicMock:
        if query_params["image"] == _TRAINING[2]:
            return _image(_TRAINING[2], "missing", "nvcr.io has no nvidia/nemo/automodel:1.")
        return _image(query_params["image"], "exists", "On the Docker host.")

    jobs = MagicMock(check_execution_profile_image=AsyncMock(side_effect=answer))
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=jobs):
        check = await check_images(_steps(_TASKS, _TRAINING, _TASKS), MagicMock())

    assert check is not None
    assert (check.passed, check.message) == (
        False,
        f"{_TRAINING[2]} is missing for gpu/default: nvcr.io has no nvidia/nemo/automodel:1.",
    )
    assert jobs.check_execution_profile_image.await_count == 2


async def test_an_image_check_the_platform_cannot_answer_passes_as_not_checked() -> None:
    request = MagicMock(status_code=404, json=MagicMock(return_value={"detail": "Not Found"}))
    jobs = MagicMock(check_execution_profile_image=AsyncMock(side_effect=NotFoundError(request)))
    with patch.object(dry_run.AsyncJobsClient, "from_client", return_value=jobs):
        check = await check_images(_steps(_TASKS), MagicMock())

    assert check is not None
    assert check.passed is True
    assert check.message.startswith(f"{_TASKS[2]}: not checked (")


def _named_check(name: str, passed: bool = True) -> JobDryRunCheck:
    return JobDryRunCheck(name=name, passed=passed, message="...")


async def _hook_checks(*, profiles_pass: bool) -> tuple[list[str], AsyncMock, AsyncMock]:
    gpus, images = AsyncMock(return_value=_named_check("GPUs")), AsyncMock(return_value=_named_check("images"))
    with (
        patch.object(
            contributor_jobs,
            "check_execution_profiles",
            AsyncMock(return_value=_named_check("execution profile", profiles_pass)),
        ),
        patch.object(contributor_jobs, "check_gpus", gpus),
        patch.object(contributor_jobs, "check_images", images),
        patch.object(
            contributor_jobs, "check_dataset_rows", AsyncMock(return_value=[_named_check("training dataset")])
        ),
    ):
        checks = await BaseSubmitJob.dry_run_checks(
            MagicMock(), platform_spec=_steps(_TRAINING), workspace="default", async_sdk=MagicMock()
        )
    return [check.name for check in checks], gpus, images


async def test_the_hook_checks_gpus_and_images_once_the_profiles_exist() -> None:
    names, gpus, images = await _hook_checks(profiles_pass=True)

    assert names == ["execution profile", "GPUs", "images", "training dataset"]
    gpus.assert_awaited_once()
    images.assert_awaited_once()


async def test_the_hook_skips_gpus_and_images_for_a_missing_profile() -> None:
    names, gpus, images = await _hook_checks(profiles_pass=False)

    assert names == ["execution profile", "training dataset"]
    gpus.assert_not_called()
    images.assert_not_called()
