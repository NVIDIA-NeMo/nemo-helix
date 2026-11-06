# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.customization_common.training.errors.parser import ParsedError, fallback_error, find_exceptions


@pytest.mark.parametrize(
    "line,expected",
    [
        ("ValueError: invalid input", ParsedError("ValueError", "invalid input")),
        (
            "[rank0]: torch.OutOfMemoryError: CUDA out of memory.",
            ParsedError("OutOfMemoryError", "CUDA out of memory."),
        ),
        ("subprocess.TimeoutExpired: Command timed out", ParsedError("TimeoutExpired", "Command timed out")),
        (
            "pyarrow.lib.ArrowInvalid: JSON parse error: Invalid encoding in string. in row 20",
            ParsedError("ArrowInvalid", "JSON parse error: Invalid encoding in string. in row 20"),
        ),
        (
            "(DTensorPolicyWorkerV2 pid=4751, ip=10.244.5.110) RuntimeError: NCCL error",
            ParsedError("RuntimeError", "NCCL error"),
        ),
        ("KeyError:", ParsedError("KeyError", "KeyError")),
    ],
)
def test_an_exception_line_gives_its_type_and_message(line, expected):
    assert find_exceptions([line]) == [expected]


@pytest.mark.parametrize(
    "line",
    [
        "frame #2: c10::DistNetworkError::DistNetworkError(c10::SourceLocation, std::string) + 0x63",
        "c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char>) + 0x98",
        "  File 'train.py', line 42, in main",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ",
        "2026-10-20 06:07:47 ERROR - Training failed with exit code: 1",
        "Error: exit status 1",
    ],
)
def test_lines_that_are_not_python_exceptions_are_ignored(line):
    assert find_exceptions([line]) == []


def test_ray_deduplication_suffix_is_removed():
    line = "(DTensorPolicyWorkerV2 pid=12) ValueError: bad sample [repeated 7x across cluster]"

    assert find_exceptions([line]) == [ParsedError("ValueError", "bad sample")]


@pytest.mark.parametrize(
    "line",
    [
        "asyncio.exceptions.CancelledError: ",
        "ImportError: sys.meta_path is None, Python is likely shutting down",
        "KeyboardInterrupt: ",
    ],
)
def test_exceptions_printed_while_shutting_down_are_ignored(line):
    assert find_exceptions([line]) == []


@pytest.mark.parametrize(
    "line",
    [
        "(raylet) Task was killed due to the node running low on memory. Memory on the node was 120GB",
        "[rank0]: tried to write checkpoint: No space left on device",
        "(VllmWorker pid=9) CUDA out of memory. Tried to allocate 1.00 GiB",
        "  traceback : Signal 9 (SIGKILL) received by PID 1061",
        'reason_message: "runtime_env_agent failed and raylet fate-shares with it."',
        'reason_message: "dashboard_agent failed and raylet fate-shares with it."',
    ],
)
def test_failures_reported_without_an_exception_line_are_kept(line):
    (parsed,) = find_exceptions([line])

    assert parsed.exception_type == "LogMessage"


def test_each_exception_is_kept_once_at_its_last_appearance():
    lines = [
        "[rank1]: FileExistsError: Checkpoint directory /c/epoch_1_step_99 already holds a published checkpoint.",
        "[rank0]: DistNetworkError: Connection reset by peer",
        "[rank0]: FileExistsError: Checkpoint directory /c/epoch_1_step_99 already holds a published checkpoint.",
    ]

    assert find_exceptions(lines) == [
        ParsedError("DistNetworkError", "Connection reset by peer"),
        ParsedError("FileExistsError", "Checkpoint directory /c/epoch_1_step_99 already holds a published checkpoint."),
    ]


def test_output_without_exceptions_falls_back_to_the_error_lines():
    lines = ["step 10 loss 1.2", "Segmentation fault (core dumped)", "done"]

    assert fallback_error(lines, 139) == ParsedError("RuntimeError", "Segmentation fault (core dumped)")


def test_the_fallback_leaves_out_config_lines_that_name_nccl():
    lines = ["backend: nccl", "step 10 loss 1.2", "Segmentation fault (core dumped)"]

    assert fallback_error(lines, 139) == ParsedError("RuntimeError", "Segmentation fault (core dumped)")


def test_empty_output_falls_back_to_the_exit_code():
    assert fallback_error([], 1) == ParsedError("RuntimeError", "Training failed with exit code: 1")


def test_a_parsed_error_rebuilds_an_exception_with_the_printed_type_name():
    exc = ParsedError("OutOfMemoryError", "CUDA out of memory.").to_exception()

    assert type(exc).__name__ == "OutOfMemoryError"
    assert str(exc) == "CUDA out of memory."
