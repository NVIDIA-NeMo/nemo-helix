# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import pytest
from nemo_customizer.jobs.atif_convert import AtifConvertJob, AtifConvertSpec, fileset_prefix, stub_output, stub_row
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job_context import JobContext
from nhx.automodel.tasks.training.datasets.schemas import SFTPChatDatasetItemSchema


def test_stub_rows_match_the_sft_chat_schema() -> None:
    first = stub_row()
    second = stub_row()
    second["messages"][1]["content"] = "What is 3+3?"
    SFTPChatDatasetItemSchema.model_validate(first)
    SFTPChatDatasetItemSchema.model_validate(second)
    output = stub_output(workspace="default", fileset="distill-data", node_key="prepare")
    assert output["dataset"] == "default/distill-data#nodes/prepare/"
    assert output["tasks"][0]["id"] == "stub-task-1"
    assert fileset_prefix("train[lora]") == "nodes/train_lora"


async def test_compile_requests_the_customizer_tasks_image(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_customizer.jobs.atif_convert as module

    spec = AtifConvertSpec(workspace="default", fileset="distill-data", node_key="prepare")
    monkeypatch.setattr(
        module,
        "_tasks_image",
        lambda: ("example.invalid/nhx-customizer-tasks:test", ["/opt/venv/bin/python"]),
    )
    compiled = await AtifConvertJob.compile(
        workspace="default",
        spec=spec,
        entity_client=None,
        job_name="distill-t1",
        async_sdk=None,
    )
    step = compiled.steps[0]
    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "example.invalid/nhx-customizer-tasks:test"
    assert step.executor.container.command == ["nemo_customizer.tasks.atif_convert"]
    assert step.config == {"workspace": "default", "fileset": "distill-data", "node_key": "prepare"}


def test_run_uploads_schema_valid_jsonl_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_customizer.jobs.atif_convert as module

    uploaded: dict[str, bytes] = {}

    class Files:
        def upload_file(self, *, workspace: str, name: str, path: str, content: bytes) -> None:
            del workspace, name
            uploaded[path] = content

    class SDK:
        pass

    monkeypatch.setattr(module.FilesClient, "from_client", lambda client: Files())
    result = AtifConvertJob().run(
        {"workspace": "default", "fileset": "distill-data", "node_key": "prepare"},
        ctx=cast(JobContext, None),
        sdk=cast(NemoClient, SDK()),
    )

    assert result["status"] == "completed"
    for path in ("nodes/prepare/train.jsonl", "nodes/prepare/validation.jsonl"):
        rows = [json.loads(line) for line in uploaded[path].decode().splitlines()]
        assert len(rows) == 2
        for row in rows:
            SFTPChatDatasetItemSchema.model_validate(row)
    output = json.loads(uploaded["nodes/prepare/output.json"])
    assert output["fileset"] == "default/distill-data"
    assert output["examples"] == 2
