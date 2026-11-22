# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from nemo_agent_hardener_plugin.api.v2.jobs import _compiler_for
from nemo_helix_plugin.jobs.api_factory import _submit_control_kwargs
from pydantic import BaseModel


class _Spec(BaseModel):
    pass


class _RecordingJob:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def compile(self, **kwargs: object) -> Mock:
        self.calls.append(kwargs)
        return Mock()


@pytest.mark.parametrize(
    "profile,options",
    [("cpu", {"priority": "high"}), (None, None)],
)
async def test_route_compiler_forwards_submit_controls(profile: str | None, options: dict | None) -> None:
    job = _RecordingJob()
    compiler = _compiler_for(job)

    # The route factory refuses these fields with a 422 when the compiler does not accept them.
    controls = _submit_control_kwargs(compiler, profile=profile, options=options)
    await compiler("default", _Spec(), _Spec(), Mock(), None, Mock(), **controls)

    assert job.calls[0]["profile"] == profile
    assert job.calls[0]["options"] == options
