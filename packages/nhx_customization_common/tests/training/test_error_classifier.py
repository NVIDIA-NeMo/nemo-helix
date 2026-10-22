# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from nhx.customization_common.training.errors.classifier import (
    NON_RESUMABLE_REASON,
    RESUMABLE,
    FailureClassifier,
    failure_status_details,
)
from nhx.customization_common.training.errors.exceptions import (
    DistributedError,
    GpuOutOfMemoryError,
    InternalError,
)

RULES = [
    {
        "id": "gpu_out_of_memory",
        "rerunnable": False,
        "exception": "GpuOutOfMemoryError",
        "contains": "CUDA out of memory",
    },
    {
        "id": "nccl_error",
        "rerunnable": True,
        "symptom": True,
        "exception": "DistributedError",
        "contains": "NCCL",
    },
]


def _classifier(tmp_path: Path, rules: list[dict] = RULES) -> FailureClassifier:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump({"rules": rules}))
    return FailureClassifier.from_yaml(path)


def test_a_cause_wins_over_a_symptom_printed_after_it(tmp_path):
    lines = [
        "[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "[rank0]: torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3866, remote process exited",
    ]

    error = _classifier(tmp_path).classify_output(lines, 1)

    assert isinstance(error, GpuOutOfMemoryError)
    assert error.cause == "gpu_out_of_memory"
    assert error.rerunnable is False
    assert "CUDA out of memory" in (error.detail or "")


def test_a_symptom_classifies_the_failure_when_nothing_else_matched(tmp_path):
    lines = ["RuntimeError: some unknown failure", "DistBackendError: NCCL communicator was aborted"]

    error = _classifier(tmp_path).classify_output(lines, 1)

    assert isinstance(error, DistributedError)
    assert error.cause == "nccl_error"
    assert error.rerunnable is True


def test_an_unmatched_failure_is_an_internal_error_that_can_be_rerun(tmp_path):
    error = _classifier(tmp_path).classify_output(["ValueError: something new"], 1)

    assert isinstance(error, InternalError)
    assert error.cause is None
    assert error.rerunnable is True
    assert "ValueError: something new" in error.message


def test_output_without_an_exception_is_classified_from_its_error_lines(tmp_path):
    error = _classifier(tmp_path).classify_output(["NCCL watchdog failed"], 1)

    assert error.cause == "nccl_error"


def test_an_in_process_exception_is_classified_by_the_exception_it_was_raised_from(tmp_path):
    try:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        except RuntimeError as oom:
            raise ValueError("training step failed") from oom
    except ValueError as e:
        error = _classifier(tmp_path).classify(e)

    assert isinstance(error, GpuOutOfMemoryError)
    assert error.rerunnable is False


def test_rules_are_checked_when_loaded(tmp_path):
    missing_id = [{"rerunnable": True, "exception": "InternalError", "contains": "x"}]
    missing_flag = [{"id": "a", "exception": "InternalError", "contains": "x"}]
    two_matchers = [{"id": "a", "rerunnable": True, "exception": "InternalError", "contains": "x", "regex": "y"}]
    unknown_field = [{"id": "a", "rerunnable": True, "exception": "InternalError", "contains": "x", "rerun": False}]
    unknown_exception = [{"id": "a", "rerunnable": True, "exception": "NoSuchError", "contains": "x"}]
    duplicate_id = [RULES[0], {**RULES[1], "id": "gpu_out_of_memory"}]

    for rules in (missing_id, missing_flag, two_matchers, unknown_field, unknown_exception, duplicate_id):
        with pytest.raises(ValueError):
            _classifier(tmp_path, rules)


def test_failure_status_details_tell_a_rerun_whether_it_would_repeat():
    oom = GpuOutOfMemoryError(message="GPU out of memory.", cause="gpu_out_of_memory", rerunnable=False)
    nccl = DistributedError(message="NCCL error.", cause="nccl_error", rerunnable=True)

    assert failure_status_details(oom) == {RESUMABLE: False, NON_RESUMABLE_REASON: "GPU out of memory."}
    assert failure_status_details(nccl) == {RESUMABLE: True}


def test_error_details_include_the_cause():
    error = GpuOutOfMemoryError(message="GPU out of memory.", detail="x", cause="gpu_out_of_memory")

    assert error.to_error_details() == {
        "message": "GPU out of memory.",
        "type": "GpuOutOfMemoryError",
        "detail": "x",
        "cause": "gpu_out_of_memory",
    }
