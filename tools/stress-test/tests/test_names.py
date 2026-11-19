# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from names import JOB_NAME_MAX_LENGTH, NameError, job_name, step_prefix


def test_job_names_are_prefixed_and_within_the_fileset_limit() -> None:
    first = job_name("stress", "exit-01", 1, 10)
    tenth = job_name("stress", "exit-10", 1, 10)
    assert first == "stress-exit-01-01"
    assert tenth == "stress-exit-10-01"
    assert not tenth.startswith(step_prefix("stress", "exit-01"))
    assert len(first) <= JOB_NAME_MAX_LENGTH


def test_job_name_over_51_characters_is_rejected() -> None:
    with pytest.raises(NameError):
        job_name("s" * 43, "exit-01", 1, 1)
