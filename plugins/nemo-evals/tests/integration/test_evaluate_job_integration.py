# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-path integration test for the row ``EvaluateJob``, focused on result persistence.

Shares the evaluator-plugin integration harness (conftest's session-scoped ``subprocess_platform``).
Submits an *offline* metric eval — inline dataset, no
model target / IGW / agent runner — so the only requirement is the host subprocess backend. Asserts the run
persisted a queryable ``EvaluateResult`` retrievable via ``Evaluator.eval_results``, covering
the row-eval half of result persistence (the agent-eval half lives in ``test_agent_evaluate_job.py``).
"""

from __future__ import annotations

import pytest
from nemo_evals.jobs.evaluate import EvaluateInputSpec, EvaluateJob
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing.e2e import wait_for_platform_job
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric

#: Opt-in: shares the evaluator-plugin integration opt-in (spins a real ``nemo services`` platform).
pytestmark = pytest.mark.integration

WORKSPACE = "default"


def _offline_exact_match_spec() -> dict:
    """An offline row-eval: a built-in metric scores inline rows that already carry expected/output.

    ExactMatch is a built-in, so it bundles inline. No target → the dataset's ``model_output`` is scored directly.
    """
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}"),
        InlineMetricBundlePackager(),
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
    client = NemoClient(base_url=subprocess_platform, retry=RetryPolicy(max_retries=2))
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
    # (Evaluator.eval_results -> the /eval-results route). Row-eval records the metric types
    # applied; an inline dataset has no dataset_ref, and an offline run has no target.
    evaluator = Evaluator.from_client(client)
    result = evaluator.eval_results.retrieve(job_name, workspace=WORKSPACE)
    assert result.job_id == job_name
    assert result.metric_types == ["exact-match"]
    assert result.dataset_ref is None
    assert result.target_kind is None
    assert result.bundle_ref
    assert result.created_at is not None

    # And it's discoverable in the workspace listing.
    listing = evaluator.eval_results.list(workspace=WORKSPACE)
    assert any(r.job_id == job_name for r in listing.data)

    # Server-side trait filtering narrows the listing (proves the SDK's filter[...] params reach the
    # entity store — the in-memory unit fakes can't exercise this).
    by_job = evaluator.eval_results.list(workspace=WORKSPACE, job_id=job_name)
    assert [r.job_id for r in by_job.data] == [job_name]
    assert evaluator.eval_results.list(workspace=WORKSPACE, job_id="no-such-job").data == []
