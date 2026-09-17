# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for task inline-metric normalization through the SDK against a real platform.

Persisting a task with an inline metric must offload that metric to a *derived* stored metric
(content-addressed, Files-backed) and leave the task holding only a reference. This exercises the
whole path end-to-end against a real entity store + Files service — the part the unit tests fake:

- an inline task metric becomes a ``default/derived.<digest>`` reference on the stored task;
- two tasks carrying byte-identical inline metrics dedupe to the *same* derived metric;
- the derived metric is real (retrievable, ``derived=True``, Files-backed);
- it is hidden from the default ``/metrics`` listing but visible with ``include_derived``.

Pure CRUD (no online target or IGW), so it only needs the host subprocess backend. Shares the evaluator-plugin
integration opt-in (``RUN_AGENT_EVAL_INTEGRATION``) and the session-scoped ``subprocess_platform``.
"""

from __future__ import annotations

import os
import uuid

import pytest
from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, MetricInline, TaskInput, TaskInputs
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
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


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _inline_metric(marker: str) -> MetricInline:
    """A valid metric whose packaged bytes are unique per ``marker`` (so the derived digest is fresh)."""
    bundle = bundle_metric(
        # The literal suffix only perturbs the template text — it keeps the metric valid while making
        # this run's content (and therefore its content-addressed derived name) distinct from others'.
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}" + marker),
        CloudpickleMetricBundlePackager(),
    )
    return MetricInline.model_validate_json(bundle.model_dump_json())


def _task_input(metric: MetricInline) -> TaskInput:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[metric],
        )
    )


@pytest.mark.timeout(300)
def test_inline_task_metric_normalizes_to_derived_metric(subprocess_platform: str) -> None:
    client = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()

    inline = _inline_metric(_unique("marker"))
    task_a = _unique("task-a")
    task_b = _unique("task-b")
    derived_name: str | None = None
    try:
        # The inline metric is offloaded: the stored task holds a single derived reference, not a bundle.
        created_a = client.evaluator.tasks.create(task_a, task=_task_input(inline), workspace=WORKSPACE)
        assert isinstance(created_a.spec, EvaluatorTaskDefinition)
        assert len(created_a.spec.metrics) == 1
        derived_ref = created_a.spec.metrics[0].root
        assert derived_ref.startswith(f"{WORKSPACE}/derived.")
        derived_name = derived_ref.split("/", 1)[1]

        # A second task with byte-identical inline content dedupes to the same derived metric.
        created_b = client.evaluator.tasks.create(task_b, task=_task_input(inline), workspace=WORKSPACE)
        assert isinstance(created_b.spec, EvaluatorTaskDefinition)
        assert created_b.spec.metrics[0].root == derived_ref

        # The derived metric is a real, Files-backed, flagged metric.
        fetched = client.evaluator.metrics.retrieve(derived_name, workspace=WORKSPACE)
        assert fetched.derived is True
        assert fetched.bundle_ref

        # Hidden from the curated default listing...
        default_names = {m.name for m in client.evaluator.metrics.list(workspace=WORKSPACE, page_size=1000).data}
        assert derived_name not in default_names

        # ...but addressable when explicitly included, exactly once (content-addressed dedup).
        with_derived = client.evaluator.metrics.list(workspace=WORKSPACE, include_derived=True, page_size=1000).data
        matching = [m for m in with_derived if m.name == derived_name]
        assert len(matching) == 1
        assert matching[0].derived is True
    finally:
        for name in (task_a, task_b):
            try:
                client.evaluator.tasks.delete(name, workspace=WORKSPACE)
            except Exception:
                pass
        if derived_name is not None:
            try:
                client.evaluator.metrics.delete(derived_name, workspace=WORKSPACE)
            except Exception:
                pass
