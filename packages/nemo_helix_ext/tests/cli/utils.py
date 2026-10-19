# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typer.testing import Result


def assert_exit_code(result: Result, expected_code: int) -> None:
    """
    Helper function to assert the exit code of a CLI command result.
    It prints stdout and stderr for easier debugging on failure.
    """
    assert result.exit_code == expected_code, (
        f"Expected exit code {expected_code}, got {result.exit_code}. stdout: {result.stdout}\nstderr: {result.stderr}"
    )
