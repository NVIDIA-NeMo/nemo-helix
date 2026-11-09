# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Gym FileSet task source against the real Files and Evals services.

Covers what submission does with a dataset file in a FileSet: the file read, the frozen job spec, and
a snapshot's stored tasks, taskset revisions and source tag, across resubmits of the same and of an
edited file. It does not run the evaluation: that needs Gym, which this host does not install, and the
run path is the ordinary inline/stored-task path covered elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import uuid

import httpx
import pytest
from nemo_evals.api.fields import MetadataItem, TaskInputs
from nemo_evals.api.schemas import EvaluatorTaskDefinition, TaskInput
from nemo_evals.api.service.task_service import stored_evaluator_task_digest
from nemo_evals.jobs.fileset_tasks import gym_reward_metric
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest

WORKSPACE = "default"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _row(question: str) -> str:
    return json.dumps({"responses_create_params": {"input": question}, "expected_answer": "A"})


class SubmitRefused(Exception):
    """The submit endpoint refused the request; the message carries its status and body."""


def _submit(platform: str, *, fileset: str, taskset: str) -> dict:
    spec = {
        "target": {
            "kind": "gym",
            "source": {
                "component": "simple_agent",
                "config": "responses_api_agents/simple_agent/configs/simple_agent.yaml",
            },
            "resources_server": "mcqa",
        },
        "tasks": {
            "source": {"fileset": f"{WORKSPACE}/{fileset}#gym/golden.jsonl", "format": "gym-jsonl"},
            "snapshot": {"taskset": f"{WORKSPACE}/{taskset}"},
        },
    }
    response = httpx.post(
        f"{platform}/apis/evals/v2/workspaces/{WORKSPACE}/agent-evaluate/jobs",
        json={"spec": spec, "profile": "default"},
        timeout=120,
    )
    if response.is_error:
        raise SubmitRefused(f"{response.status_code}: {response.text}")
    name = response.json().get("name")
    assert name, f"submit response carried no job name: {response.text}"
    return (
        httpx.get(f"{platform}/apis/evals/v2/workspaces/{WORKSPACE}/agent-evaluate/jobs/{name}")
        .raise_for_status()
        .json()["spec"]
    )


@pytest.mark.timeout(300)
def test_snapshots_track_the_golden_file_across_resubmits(subprocess_platform: str) -> None:
    client = NemoClient(base_url=subprocess_platform, workspace=WORKSPACE, retry=RetryPolicy(max_retries=2))
    WorkspacesClient.from_client(client).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    files = FilesClient.from_client(client)
    evaluator = Evaluator.from_client(client)
    fileset, taskset = _unique("golden"), _unique("golden-suite")
    files.create_fileset(workspace=WORKSPACE, body=CreateFilesetRequest(name=fileset))

    def upload(*lines: str) -> str:
        body = ("\n".join(lines) + "\n").encode()
        files.upload_file(workspace=WORKSPACE, name=fileset, path="gym/golden.jsonl", content=body)
        return hashlib.sha256(body).hexdigest()

    # First version: every row is published, and the job runs exactly that revision.
    sha_v1 = upload(_row("q1"), _row("q2"))
    first = _submit(subprocess_platform, fileset=fileset, taskset=taskset)
    source = first["task_source"]
    assert source["file_sha256"] == sha_v1
    assert source["task_count"] == 2
    assert source["snapshot"]["delta"] == {"added": 2, "changed": 0, "removed": 0, "unchanged": 0}
    pinned = source["snapshot"]["taskset"]
    digest = pinned.rsplit("#", 1)[1]
    stored = evaluator.tasksets.retrieve(taskset, revision=digest, workspace=WORKSPACE)
    assert stored.content_hash == digest
    assert f"src-{sha_v1}" in stored.tags
    for member, task in zip(stored.tasks, first["tasks"], strict=True):
        provenance = task["spec"]["provenance"]
        assert member.root == f"{provenance['entity_name']}#{provenance['revision_digest']}"
        assert task["spec"]["intent"] == "Gym row"

    # The same bytes again: nothing new is published, and the job runs the same revision.
    second = _submit(subprocess_platform, fileset=fileset, taskset=taskset)
    assert second["task_source"]["snapshot"]["taskset"] == pinned
    assert second["task_source"]["snapshot"]["delta"]["unchanged"] == 2
    revisions = evaluator.tasksets.list_revisions(taskset, workspace=WORKSPACE)
    assert len(revisions.data) == 1

    # An edited row: one revision more, the old task kept but no longer a member.
    sha_v2 = upload(_row("q1"), _row("q2 edited"))
    third = _submit(subprocess_platform, fileset=fileset, taskset=taskset)
    assert third["task_source"]["file_sha256"] == sha_v2
    assert third["task_source"]["snapshot"]["delta"] == {"added": 1, "changed": 0, "removed": 1, "unchanged": 1}
    assert len(evaluator.tasksets.list_revisions(taskset, workspace=WORKSPACE).data) == 2
    dropped = {task["id"] for task in first["tasks"]} - {task["id"] for task in third["tasks"]}
    assert len(dropped) == 1
    assert evaluator.tasks.retrieve(dropped.pop(), workspace=WORKSPACE).name


@pytest.mark.timeout(300)
def test_a_bad_row_is_refused_and_writes_nothing(subprocess_platform: str) -> None:
    client = NemoClient(base_url=subprocess_platform, workspace=WORKSPACE, retry=RetryPolicy(max_retries=2))
    WorkspacesClient.from_client(client).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    files = FilesClient.from_client(client)
    fileset, taskset = _unique("golden"), _unique("golden-suite")
    files.create_fileset(workspace=WORKSPACE, body=CreateFilesetRequest(name=fileset))
    files.upload_file(
        workspace=WORKSPACE, name=fileset, path="gym/golden.jsonl", content=(_row("q1") + "\n{broken\n").encode()
    )

    with pytest.raises(SubmitRefused, match=r"422: .*golden\.jsonl:2: malformed JSON"):
        _submit(subprocess_platform, fileset=fileset, taskset=taskset)
    response = httpx.get(f"{subprocess_platform}/apis/evals/v2/workspaces/{WORKSPACE}/tasksets/{taskset}")
    assert response.status_code == 404


@pytest.mark.timeout(300)
def test_the_computed_digest_is_the_one_the_service_publishes(subprocess_platform: str) -> None:
    """A snapshot skips writing a row whose computed digest the previous revision pins; if the
    computation drifted from the service, every resubmit would rewrite every row."""
    client = NemoClient(base_url=subprocess_platform, workspace=WORKSPACE, retry=RetryPolicy(max_retries=2))
    evaluator = Evaluator.from_client(client)
    task = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Gym row",
            inputs=TaskInputs(gym_row={"input": "q"}),
            metrics=[gym_reward_metric()],
        ),
        metadata=[MetadataItem(key="gym_row_extras", value={"expected_answer": "A"})],
    )
    stored = evaluator.tasks.replace(_unique("gym"), task=task, workspace=WORKSPACE)
    assert stored.content_hash == stored_evaluator_task_digest(task, workspace=WORKSPACE)
