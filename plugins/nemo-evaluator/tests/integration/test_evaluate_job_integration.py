# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-path integration test for the row ``EvaluateJob``, focused on result persistence.

Shares the evaluator-plugin integration harness (conftest's session-scoped ``subprocess_platform``)
and the ``RUN_AGENT_EVAL_INTEGRATION`` opt-in. Submits an *offline* metric eval — inline dataset, no
model target / IGW / agent runner — so the only requirement is the host subprocess backend. Asserts the run
persisted a queryable ``EvaluateResult`` retrievable via ``client.evaluator.eval_results``, covering
the row-eval half of result persistence (the agent-eval half lives in ``test_agent_evaluate_job.py``).
"""

from __future__ import annotations

import os

import pytest
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec, EvaluateJob
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.scheduler import NemoJobScheduler
from nemo_platform_plugin.sdk import NeMoPlatform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.testing.e2e import wait_for_platform_job

#: Opt-in: shares the evaluator-plugin integration opt-in (spins a real ``nemo services`` platform).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]

WORKSPACE = "default"


def _offline_exact_match_spec() -> dict:
    """An offline row-eval: a built-in metric scores inline rows that already carry expected/output.

    ExactMatch is a built-in (importable in the submit-backend subprocess), so a cloudpickle bundle
    round-trips fine there. No target → the dataset's ``model_output`` is scored directly.
    """
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
        CloudpickleMetricBundlePackager(),
    )
    return EvaluateInputSpec.model_validate(
        {
            "metrics": [bundle.model_dump(mode="json")],
            "dataset": [
                {"expected": "blue", "model_output": "blue"},
                {"expected": "Jupiter", "model_output": "Jupiter"},
            ],
        }
    ).model_dump(mode="json")


@pytest.mark.timeout(600)
def test_submit_offline_row_eval_persists_result(subprocess_platform: str) -> None:
    # dim: submit x subprocess backend, row (EvaluateJob) path. The jobs service compiles + runs
    # EvaluateJob.run() as a host subprocess; run() writes an EvaluateResult through the async task
    # SDK + entity store. Offline (no target or IGW): the dataset already carries the outputs.
    client = NeMoPlatform(base_url=subprocess_platform, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()

    response = NemoJobScheduler().submit_remote(
        EvaluateJob, _offline_exact_match_spec(), base_url=subprocess_platform, workspace=WORKSPACE, profile="default"
    )
    job_name = response.get("name") or response.get("id")
    assert job_name, f"submit response carried no job name/id: {response}"

    job = wait_for_platform_job(client, job_name, WORKSPACE, timeout=480)
    assert job.status == "completed", f"job {job_name} ended {job.status!r}: {getattr(job, 'status_details', None)}"

    # Persistence: run() wrote a queryable EvaluateResult, retrievable via the typed SDK resource
    # (client.evaluator.eval_results -> the /eval-results route). Row-eval records the metric types
    # applied; an inline dataset has no dataset_ref, and an offline run has no target.
    result = client.evaluator.eval_results.retrieve(job_name, workspace=WORKSPACE)
    assert result.job_id == job_name
    assert result.metric_types == ["exact-match"]
    assert result.dataset_ref is None
    assert result.target_kind is None
    assert result.bundle_ref
    assert result.created_at is not None

    # And it's discoverable in the workspace listing.
    listing = client.evaluator.eval_results.list(workspace=WORKSPACE)
    assert any(r.job_id == job_name for r in listing.data)

    # Server-side trait filtering narrows the listing (proves the SDK's filter[...] params reach the
    # entity store — the in-memory unit fakes can't exercise this).
    by_job = client.evaluator.eval_results.list(workspace=WORKSPACE, job_id=job_name)
    assert [r.job_id for r in by_job.data] == [job_name]
    assert client.evaluator.eval_results.list(workspace=WORKSPACE, job_id="no-such-job").data == []
