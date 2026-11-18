# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub ATIF convert job.

Writes a schema-valid ``train.jsonl``, ``validation.jsonl`` and ``output.json``
into the workflow fileset. The rows are synthetic. A later pass replaces this
with the real trajectory converter; the fileset layout stays the same.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from pydantic import BaseModel, ConfigDict

SOURCE = "customization"


class AtifConvertSpec(BaseModel):
    """What the workflow controller hands the stub."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    fileset: str
    node_key: str


def fileset_prefix(node_key: str) -> str:
    """Fileset directory for one node. Brackets in matrix keys are not path-safe."""
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in node_key).strip("_")
    if not safe:
        raise ValueError(f"node key {node_key!r} has no path-safe characters")
    return f"nodes/{safe}"


def stub_row() -> dict[str, Any]:
    """One ``SFTChatMessage`` conversation: system, user, thinking, tool call, tool result, reply."""
    return {
        "messages": [
            {"role": "system", "content": "You are a stub teacher. This row is plumbing, not a real trajectory."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "thinking": "Add the two numbers."},
            {
                "role": "assistant",
                "content": "I will calculate that.",
                "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expression": "2+2"}}}],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def stub_output(*, workspace: str, fileset: str, node_key: str) -> dict[str, Any]:
    """The ``output.json`` the workflow controller reads after this job completes."""
    prefix = fileset_prefix(node_key)
    return {
        "fileset": f"{workspace}/{fileset}",
        "dataset": f"{workspace}/{fileset}#{prefix}/",
        "held_out": ["stub-task-1"],
        "runs": 1,
        "examples": 2,
        "tasks": [{"id": "stub-task-1", "intent": "Stub task for distillation plumbing."}],
    }


def _jsonl(row: dict[str, Any]) -> bytes:
    return (json.dumps(row) + "\n").encode("utf-8")


def _tasks_image() -> tuple[str, list[str]]:
    from nhx.customization_common.service.images import (
        CUSTOMIZER_PYTHON_ENTRYPOINT,
        get_customizer_tasks_image,
    )

    return get_customizer_tasks_image(), CUSTOMIZER_PYTHON_ENTRYPOINT


class AtifConvertJob(NemoJob):
    """Upload the stub dataset into the workflow fileset and return success."""

    name: ClassVar[str] = "atif-convert"
    description: ClassVar[str] = "Write a synthetic training fileset for a distillation workflow."
    container: ClassVar[str] = "nhx-customizer-tasks"
    spec_schema: ClassVar[type[BaseModel] | None] = AtifConvertSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HelixJobSpec:
        del workspace, entity_client, job_name, async_sdk, options
        image, entrypoint = _tasks_image()
        return HelixJobSpec(
            steps=[
                HelixJobStepSpec(
                    name="convert",
                    executor=CPUExecutionProvider(
                        provider="cpu",
                        profile=profile or "default",
                        container=ContainerSpec(
                            image=image,
                            entrypoint=[*entrypoint, "-m"],
                            command=["nemo_customizer.tasks.atif_convert"],
                        ),
                    ),
                    config=spec.model_dump(mode="json"),
                    environment=[
                        HelixJobEnvironmentVariable(
                            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
                            value=DEFAULT_JOB_STORAGE_PATH,
                        )
                    ],
                )
            ]
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient | None = None) -> dict[str, Any]:
        del ctx
        spec = AtifConvertSpec.model_validate(config)
        if sdk is None:
            raise RuntimeError("atif-convert requires a platform client to upload the workflow fileset")
        prefix = fileset_prefix(spec.node_key)
        rows = [stub_row(), stub_row()]
        rows[1]["messages"][1]["content"] = "What is 3+3?"
        payload = b"".join(_jsonl(row) for row in rows)
        output = stub_output(workspace=spec.workspace, fileset=spec.fileset, node_key=spec.node_key)
        files = FilesClient.from_client(sdk)
        files.upload_file(workspace=spec.workspace, name=spec.fileset, path=f"{prefix}/train.jsonl", content=payload)
        files.upload_file(
            workspace=spec.workspace,
            name=spec.fileset,
            path=f"{prefix}/validation.jsonl",
            content=payload,
        )
        files.upload_file(
            workspace=spec.workspace,
            name=spec.fileset,
            path=f"{prefix}/output.json",
            content=json.dumps(output).encode("utf-8"),
        )
        return {"status": "completed", "dataset": output["dataset"]}
