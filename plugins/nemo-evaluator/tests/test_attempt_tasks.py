# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from nemo_evaluator.jobs.attempt_tasks import (
    TASKS_DIGEST_KEY,
    TASKS_FROM_ATTEMPT,
    TASKS_FROM_KEY,
    AttemptTasksError,
    load_attempt_tasks,
    step_config_without_tasks,
)
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths


def test_step_config_without_tasks_and_round_trip() -> None:
    spec = {"tasks": [{"id": "t1"}], "trials": [{"id": "tr1"}], "fail_fast": False}
    stripped = step_config_without_tasks(spec)
    assert "tasks" not in stripped and "trials" not in stripped
    assert stripped[TASKS_FROM_KEY] == TASKS_FROM_ATTEMPT
    job_doc = {
        "id": "j-id",
        "attempt_id": "a-1",
        "name": "job-1",
        "workspace": "default",
        "source": "evaluator",
        "spec": spec,
        "platform_spec": {"steps": []},
        "fileset": "fs-1",
        "status": "active",
    }
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=job_doc))
    http_client = httpx.Client(transport=transport, base_url="http://platform.local")
    client = NemoClient(base_url="http://platform.local", http_client=http_client)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = JobContext(
            workspace="default",
            storage=StoragePaths(Path(tmp)),
            results=SimpleNamespace(),
            job_id="job-1",
        )
        restored = load_attempt_tasks(stripped, ctx=ctx, client=client)
        assert restored == spec
        with pytest.raises(AttemptTasksError):
            load_attempt_tasks({**stripped, TASKS_DIGEST_KEY: "sha256-v1:" + "0" * 64}, ctx=ctx, client=client)
