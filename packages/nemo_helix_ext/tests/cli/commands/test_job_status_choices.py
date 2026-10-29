# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import get_args

from nemo_helix_ext.cli.commands.jobs import JobStatusValue
from nemo_helix_plugin.jobs.schemas import HelixJobStatus


def test_cli_status_choices_cover_every_job_status():
    assert set(get_args(JobStatusValue)) == {status.value for status in HelixJobStatus}


def test_cli_status_choices_include_queued():
    assert "queued" in get_args(JobStatusValue)
