# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``Manage Tasks & Tasksets`` doc walkthrough, executed against a real platform.

``make docs-check-python-snippets`` type-checks the doc's snippets, which catches a snippet that
names a field that no longer exists — but not one that type-checks and then fails at run time, and
not a documented *output* that no longer matches. Both happened: the task model moved its content
under a discriminated ``spec``, and the revision snippets in this doc kept the old flat shape
through review because nothing executed them.

So this walks the doc top to bottom, in order, doing what it says and asserting the results it
claims. It deliberately mirrors the doc's own code rather than being written as an idiomatic test —
when it fails, the fix is usually the doc.

Pure CRUD (no online target or IGW), so it only needs the host subprocess backend. Shares the evaluator-plugin
integration opt-in (``RUN_AGENT_EVAL_INTEGRATION``) and the session-scoped ``subprocess_platform``.
"""

from __future__ import annotations

import os
import uuid

import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetadataItem,
    MetricRef,
    TaskInput,
    TaskInputs,
    TaskRef,
    TasksetInput,
    TasksetRef,
)
from nemo_evaluator_sdk import ExactMatchMetric
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.sdk import NeMoPlatform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]

WORKSPACE = "default"


@pytest.fixture
def doc_client(subprocess_platform: str) -> NeMoPlatform:
    """The doc's own ``Initialize the SDK`` snippet, with the base URL the fixture provides.

    ``workspace=`` on the constructor is part of what is being checked: every later snippet omits a
    per-call workspace and relies on this default.
    """
    client = NeMoPlatform(base_url=subprocess_platform, workspace=WORKSPACE, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    return client


def _unique(prefix: str) -> str:
    """Names are per-test so a reused platform can't leak state between them."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.timeout(300)
def test_the_manage_tasks_walkthrough(doc_client: NeMoPlatform) -> None:
    """``Manage Tasks`` through ``Tag a revision`` — create, read, publish, pin, tag."""
    client = doc_client
    tasks = client.evaluator.tasks
    task_name = _unique("capital-of-france")
    metric_name = _unique("answer-exact-match")

    client.evaluator.metrics.create(
        metric_name,
        metric=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
    )

    task = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the user's geography question with the capital city.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            metrics=[MetricRef(f"{WORKSPACE}/{metric_name}")],
        ),
        metadata=[MetadataItem(key="suite", value="geography")],
    )

    stored = tasks.create(task_name, task=task)
    # The doc prints `stored.id, stored.spec.metrics` and states that a stored task holds metric
    # *references* only.
    assert stored.id
    assert [ref.root for ref in stored.spec.metrics] == [f"{WORKSPACE}/{metric_name}"]

    # "Retrieve, list, and delete" — the doc's comment claims `evaluator 1 {'latest': 1}`.
    retrieved = tasks.retrieve(task_name)
    assert (retrieved.spec.kind, retrieved.revision, retrieved.tags) == ("evaluator", 1, {"latest": 1})

    page = tasks.list(page=1, page_size=100, sort="-created_at")
    assert (task_name, "evaluator") in [(item.name, item.spec.kind) for item in page.data]

    # "Publish a new revision" — the doc's comment claims revision 2.
    revised_task = TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the user's geography question with the capital city.",
            inputs=TaskInputs(instruction="Name the capital city of France."),
            metrics=[MetricRef(f"{WORKSPACE}/{metric_name}")],
        ),
        metadata=[MetadataItem(key="suite", value="geography")],
    )
    updated = tasks.replace(task_name, task=revised_task)
    assert updated.revision == 2

    # The doc's idempotence Note: re-submitting identical content publishes nothing.
    assert tasks.replace(task_name, task=revised_task).revision == 2

    # "Read a specific revision" — a pinned read returns what was published, not what is current.
    revisions = tasks.list_revisions(task_name)
    digest = next(revision.content_hash for revision in revisions.data if revision.revision == 1)

    original = tasks.retrieve(task_name, revision=digest)
    current = tasks.retrieve(task_name)
    assert original.revision == 1 and current.revision == 2
    assert original.spec.inputs.instruction != current.spec.inputs.instruction

    # "Tag a revision", including the documented `ValueError` when both selectors are passed.
    tasks.tag(task_name, tag="blessed", revision=digest)
    blessed = tasks.retrieve(task_name, tag="blessed")
    assert blessed.revision == 1
    with pytest.raises(ValueError):
        tasks.retrieve(task_name, revision=digest, tag="blessed")

    tasks.delete(task_name)


@pytest.mark.timeout(300)
def test_the_manage_tasksets_walkthrough(doc_client: NeMoPlatform) -> None:
    """``Manage Tasksets`` and ``Pin the taskset itself`` — membership pinning is the claim."""
    client = doc_client
    tasks = client.evaluator.tasks
    tasksets = client.evaluator.tasksets
    france, japan = _unique("capital-of-france"), _unique("capital-of-japan")
    suite = _unique("geography-suite")

    for name, city in ((france, "France"), (japan, "Japan")):
        tasks.create(
            name,
            task=TaskInput(
                spec=EvaluatorTaskDefinition(
                    kind="evaluator",
                    intent="Answer the user's geography question with the capital city.",
                    inputs=TaskInputs(instruction=f"What is the capital of {city}?"),
                )
            ),
        )

    taskset = TasksetInput(
        description="Geography questions for smoke-testing the agent.",
        tasks=[TaskRef(f"{WORKSPACE}/{france}"), TaskRef(f"{WORKSPACE}/{japan}")],
    )
    stored = tasksets.create(suite, taskset=taskset)

    # The doc's central claim: a bare member ref is stored resolved to `workspace/name#<digest>`.
    assert all("#" in ref.root for ref in stored.tasks)
    assert {ref.root.split("#")[0] for ref in stored.tasks} == {f"{WORKSPACE}/{france}", f"{WORKSPACE}/{japan}"}

    page = tasksets.list(page=1, page_size=100, sort="name")
    assert suite in [item.name for item in page.data]
    assert tasksets.retrieve(suite).description == "Geography questions for smoke-testing the agent."

    # The doc's Note: member *order* is not part of a taskset's identity, so reordering the same
    # members publishes nothing.
    reordered = TasksetInput(
        description="Geography questions for smoke-testing the agent.",
        tasks=[TaskRef(f"{WORKSPACE}/{japan}"), TaskRef(f"{WORKSPACE}/{france}")],
    )
    assert tasksets.replace(suite, taskset=reordered).revision == 1

    # ...but re-resolving after a member republishes genuinely differs, so it does cut a revision.
    tasks.replace(
        france,
        task=TaskInput(
            spec=EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Answer the user's geography question with the capital city.",
                inputs=TaskInputs(instruction="Name the capital city of France."),
            )
        ),
    )
    assert tasksets.replace(suite, taskset=taskset).revision == 2

    # "Pin the taskset itself" — both ref forms are accepted by the field.
    current = tasksets.list_revisions(suite).data[0]
    assert current.revision == 2  # revisions come back newest-first, as the doc's comment says
    assert TasksetRef(f"{WORKSPACE}/{suite}").root
    assert TasksetRef(f"{WORKSPACE}/{suite}#{current.content_hash}").root

    # Deleting a taskset does not delete its member tasks.
    tasksets.delete(suite)
    assert tasks.retrieve(france).name == france

    for name in (france, japan):
        tasks.delete(name)
