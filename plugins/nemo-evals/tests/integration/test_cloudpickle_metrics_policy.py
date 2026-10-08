# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cloudpickle metric opt-in, end to end against real platforms.

Every metric here is a :class:`_RecordingMetric`, which appends a line to a host file each time any
process unpickles it. That turns "the platform never deserialized this payload" into an assertion
on a file, across the API process and the job subprocesses alike.

``subprocess_platform`` is opted in (API config and executor env). ``cloudpickle_toggle_platform``
is seeded opted in, then restarted opted out, the way an operator would disable cloudpickle metrics
on a deployment that already stored some.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import httpx
import pytest
from nemo_evals.api.schemas import EvaluatorTaskDefinition, MetricInline, TaskInput, TaskInputs, TaskRef
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, AgentEvalTaskInput
from nemo_evals.jobs.evaluate import EvaluateInputSpec, EvaluateJob
from nemo_evals.metric_refs import MetricRef
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values import Model, SecretRef
from nhx_evals_sdk.values.common import SupportedJobTypes
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.scheduler import submit_path_for
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing.e2e import wait_for_platform_job

pytestmark = pytest.mark.integration

WORKSPACE = "default"
DISABLED_MESSAGE = "cloudpickle metrics are disabled on this deployment"
EVALUATOR_API = f"/apis/evals/v2/workspaces/{WORKSPACE}"

# The job subprocesses cannot import this module, so the metric class must travel inside the pickle.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _RecordingMetric:
    """Scores 1.0 on every row, and appends a line to ``marker`` whenever it is unpickled."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        with open(self.marker, "a", encoding="utf-8") as handle:
            handle.write("unpickled\n")

    @property
    def type(self) -> str:
        return "recording"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # noqa: A002
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _unpickle_count(marker: Path) -> int:
    return len(marker.read_text(encoding="utf-8").splitlines()) if marker.exists() else 0


def _cloudpickle_metric(marker: Path) -> dict:
    bundle = bundle_metric(_RecordingMetric(str(marker)), CloudpickleMetricBundlePackager())
    return MetricInline.model_validate_json(bundle.model_dump_json()).model_dump(mode="json")


def _inline_exact_match() -> dict:
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"), InlineMetricBundlePackager()
    )
    return MetricInline.model_validate_json(bundle.model_dump_json()).model_dump(mode="json")


def _evaluate_spec(metric: dict | str) -> dict:
    return EvaluateInputSpec.model_validate(
        {"metrics": [metric], "dataset": [{"expected": "blue", "output": "blue"}]}
    ).model_dump(mode="json")


def _agent_eval_spec(metric: dict) -> dict:
    """An offline agent eval: one task carrying ``metric``, scored against one precomputed trial."""
    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="say-done",
                intent="Agent replies DONE.",
                inputs=TaskInputs(instruction="Reply with DONE."),
                metrics=[MetricInline.model_validate(metric)],
            )
        ],
        trials=[
            AgentEvalTrial(
                id="t-1",
                task_id="say-done",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="DONE"),
            )
        ],
    ).model_dump(mode="json")


def _task_body(metric: dict | str) -> dict:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef(metric) if isinstance(metric, str) else MetricInline.model_validate(metric)],
        )
    ).model_dump(mode="json")


def _live_body(metric: dict | str) -> dict:
    return {
        "dataset": [{"expected": "blue", "output": "blue"}],
        "metrics": [metric],
        "field_mapping": {"output": "output"},
    }


def _submit(http: httpx.Client, job_cls: type[NemoJob], spec: dict, *, profile: str = "default") -> httpx.Response:
    return http.post(submit_path_for(job_cls, workspace=WORKSPACE), json={"spec": spec, "profile": profile})


def _client(base_url: str) -> NemoClient:
    client = NemoClient(base_url=base_url, workspace=WORKSPACE, retry=RetryPolicy(max_retries=2))
    WorkspacesClient.from_client(client).create_workspace(exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE))
    return client


def _run_to_completion(
    base_url: str, http: httpx.Client, job_cls: type[NemoJob], spec: dict, *, profile: str = "default"
):
    response = _submit(http, job_cls, spec, profile=profile)
    assert response.status_code in (200, 201), response.text
    name = response.json()["name"]
    return name, wait_for_platform_job(_client(base_url), name, WORKSPACE, timeout=300)


def _task_log(job) -> str:
    """The job's task logs, read from the subprocess backend's work dir on this host."""
    work_dir = Path(job.status_details["subprocess_work_dir"])
    return "\n".join(path.read_text(encoding="utf-8") for path in work_dir.rglob("*.log*"))


# ---- opted in -----------------------------------------------------------------------------------


@pytest.fixture
def opted_in(subprocess_platform: str) -> Iterator[httpx.Client]:
    _client(subprocess_platform)
    with httpx.Client(base_url=subprocess_platform, timeout=120) as http:
        yield http


@pytest.mark.timeout(600)
def test_opted_in_evaluate_job_scores_inline_cloudpickle_metric_and_reads_never_unpickle_it(
    subprocess_platform: str, opted_in: httpx.Client, tmp_path: Path
) -> None:
    marker = tmp_path / "unpickled.log"

    name, job = _run_to_completion(
        subprocess_platform, opted_in, EvaluateJob, _evaluate_spec(_cloudpickle_metric(marker))
    )

    assert job.status == "completed", job.status_details
    unpickled_by_submit_and_run = _unpickle_count(marker)
    assert unpickled_by_submit_and_run > 0
    got = opted_in.get(f"{EVALUATOR_API}/evaluate/jobs/{name}")
    listed = opted_in.get(f"{EVALUATOR_API}/evaluate/jobs")
    assert got.status_code == 200, got.text
    assert listed.status_code == 200, listed.text
    assert got.json()["spec"]["metrics"][0]["payload"]["kind"] == "cloudpickle"
    assert _unpickle_count(marker) == unpickled_by_submit_and_run


@pytest.mark.timeout(600)
def test_opted_in_evaluate_job_scores_stored_cloudpickle_metric(
    subprocess_platform: str, opted_in: httpx.Client, tmp_path: Path
) -> None:
    metric_name = _unique("custom")
    created = opted_in.post(f"{EVALUATOR_API}/metrics/{metric_name}", json=_cloudpickle_metric(tmp_path / "m.log"))
    assert created.status_code == 201, created.text
    assert created.json()["payload_kind"] == "cloudpickle"

    _, job = _run_to_completion(
        subprocess_platform, opted_in, EvaluateJob, _evaluate_spec(f"{WORKSPACE}/{metric_name}")
    )

    assert job.status == "completed", job.status_details


@pytest.mark.timeout(600)
def test_opted_in_agent_eval_scores_task_with_inline_cloudpickle_metric(
    subprocess_platform: str, opted_in: httpx.Client, tmp_path: Path
) -> None:
    task_name = _unique("custom-task")
    created = opted_in.post(
        f"{EVALUATOR_API}/tasks/{task_name}", json=_task_body(_cloudpickle_metric(tmp_path / "m.log"))
    )
    assert created.status_code == 201, created.text
    spec = _agent_eval_spec(_cloudpickle_metric(tmp_path / "m.log"))
    spec["tasks"] = [TaskRef(f"{WORKSPACE}/{task_name}").model_dump(mode="json")]
    spec["trials"][0]["task_id"] = task_name

    _, job = _run_to_completion(subprocess_platform, opted_in, AgentEvalJob, spec)

    assert job.status == "completed", job.status_details


def test_opted_in_live_scores_cloudpickle_metric(opted_in: httpx.Client, tmp_path: Path) -> None:
    response = opted_in.post(f"{EVALUATOR_API}/evaluate/live", json=_live_body(_cloudpickle_metric(tmp_path / "m.log")))

    assert response.status_code == 200, response.text
    assert response.json()["metrics"][0]["scores"][0]["mean"] == pytest.approx(1.0)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "job_cls,spec_for",
    [(EvaluateJob, _evaluate_spec), (AgentEvalJob, _agent_eval_spec)],
    ids=["evaluate", "agent-evaluate"],
)
def test_worker_refuses_cloudpickle_metric_when_its_executor_is_not_opted_in(
    subprocess_platform: str,
    opted_in: httpx.Client,
    tmp_path: Path,
    cloudpickle_off_profile: str,
    job_cls: type[NemoJob],
    spec_for: Callable[[dict], dict],
) -> None:
    """The API accepts the job, so only the worker's own check stands between the payload and its unpickling."""
    marker = tmp_path / "unpickled.log"
    spec = spec_for(_cloudpickle_metric(marker))

    _, job = _run_to_completion(subprocess_platform, opted_in, job_cls, spec, profile=cloudpickle_off_profile)

    assert job.status == "error", job.status_details
    assert DISABLED_MESSAGE in _task_log(job)


def test_submitter_cannot_grant_their_worker_the_opt_in_through_a_secret(
    opted_in: httpx.Client, tmp_path: Path
) -> None:
    """A judge's key env name derives from its secret's name, and config matches env names case-insensitively."""
    secret = "nemo-evals-allow-insecure-cloudpickle-metrics"
    created = opted_in.post(f"/apis/secrets/v2/workspaces/{WORKSPACE}/secrets", json={"name": secret, "value": "true"})
    assert created.status_code in (200, 201, 409), created.text
    judge = LLMJudgeMetric(
        model=Model(
            url="http://judge.invalid/v1/chat/completions", name="judge", api_key_secret=SecretRef(root=secret)
        ),
        scores=[RangeScore(name="q", minimum=0, maximum=1, parser=JSONScoreParser(json_path="q"))],
        job_type=SupportedJobTypes.OFFLINE,
    )
    judge_metric = MetricInline.model_validate_json(
        bundle_metric(judge, InlineMetricBundlePackager()).model_dump_json()
    ).model_dump(mode="json")
    spec = EvaluateInputSpec.model_validate(
        {"metrics": [_cloudpickle_metric(tmp_path / "m.log"), judge_metric], "dataset": [{"output": "blue"}]}
    ).model_dump(mode="json")

    response = _submit(opted_in, EvaluateJob, spec, profile="cloudpickle-off")

    assert response.status_code == 422, response.text
    assert "'nemo_evals_allow_insecure_cloudpickle_metrics' is reserved" in response.text


# ---- opted out after storing cloudpickle metrics --------------------------------------------------


@dataclass(frozen=True)
class _OptedOut:
    http: httpx.Client
    base_url: str
    marker: Path
    stored_metric: str
    stored_task: str
    stored_job: str


@pytest.fixture(scope="module")
def opted_out(
    cloudpickle_toggle_platform: Callable[..., AbstractContextManager[str]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_OptedOut]:
    marker = tmp_path_factory.mktemp("cloudpickle-policy") / "unpickled.log"
    metric = _cloudpickle_metric(marker)
    stored_metric, stored_task = _unique("legacy-metric"), _unique("legacy-task")
    with cloudpickle_toggle_platform(allow_cloudpickle_metrics=True) as base_url:
        _client(base_url)
        with httpx.Client(base_url=base_url, timeout=120) as http:
            assert http.post(f"{EVALUATOR_API}/metrics/{stored_metric}", json=metric).status_code == 201
            assert http.post(f"{EVALUATOR_API}/tasks/{stored_task}", json=_task_body(metric)).status_code == 201
            stored_job, job = _run_to_completion(base_url, http, EvaluateJob, _evaluate_spec(metric))
            assert job.status == "completed", job.status_details
    marker.unlink(missing_ok=True)

    with cloudpickle_toggle_platform(allow_cloudpickle_metrics=False) as base_url:
        with httpx.Client(base_url=base_url, timeout=120) as http:
            yield _OptedOut(http, base_url, marker, stored_metric, stored_task, stored_job)


def _assert_refused(response: httpx.Response, opted_out: _OptedOut) -> None:
    assert response.status_code == 422, response.text
    assert DISABLED_MESSAGE in response.text
    assert _unpickle_count(opted_out.marker) == 0


def test_opted_out_metric_create_is_refused(opted_out: _OptedOut) -> None:
    name = _unique("custom")

    _assert_refused(
        opted_out.http.post(f"{EVALUATOR_API}/metrics/{name}", json=_cloudpickle_metric(opted_out.marker)), opted_out
    )
    assert opted_out.http.get(f"{EVALUATOR_API}/metrics/{name}").status_code == 404


@pytest.mark.parametrize("method", ["post", "put"])
def test_opted_out_task_with_inline_cloudpickle_metric_is_refused(opted_out: _OptedOut, method: str) -> None:
    name = _unique("custom-task")

    response = opted_out.http.request(
        method.upper(), f"{EVALUATOR_API}/tasks/{name}", json=_task_body(_cloudpickle_metric(opted_out.marker))
    )

    _assert_refused(response, opted_out)
    assert opted_out.http.get(f"{EVALUATOR_API}/tasks/{name}").status_code == 404


def test_opted_out_evaluate_job_with_inline_cloudpickle_metric_is_refused(opted_out: _OptedOut) -> None:
    _assert_refused(
        _submit(opted_out.http, EvaluateJob, _evaluate_spec(_cloudpickle_metric(opted_out.marker))), opted_out
    )


def test_opted_out_evaluate_job_referencing_stored_cloudpickle_metric_is_refused(opted_out: _OptedOut) -> None:
    _assert_refused(
        _submit(opted_out.http, EvaluateJob, _evaluate_spec(f"{WORKSPACE}/{opted_out.stored_metric}")), opted_out
    )


def test_opted_out_agent_eval_with_inline_cloudpickle_metric_is_refused(opted_out: _OptedOut) -> None:
    _assert_refused(
        _submit(opted_out.http, AgentEvalJob, _agent_eval_spec(_cloudpickle_metric(opted_out.marker))), opted_out
    )


def test_opted_out_agent_eval_over_stored_task_with_cloudpickle_metric_is_refused(opted_out: _OptedOut) -> None:
    spec = _agent_eval_spec(_inline_exact_match())
    spec["tasks"] = [TaskRef(f"{WORKSPACE}/{opted_out.stored_task}").model_dump(mode="json")]
    spec["trials"][0]["task_id"] = opted_out.stored_task

    _assert_refused(_submit(opted_out.http, AgentEvalJob, spec), opted_out)


@pytest.mark.parametrize("stored", [False, True], ids=["inline", "stored-ref"])
def test_opted_out_live_with_cloudpickle_metric_is_refused(opted_out: _OptedOut, stored: bool) -> None:
    metric = f"{WORKSPACE}/{opted_out.stored_metric}" if stored else _cloudpickle_metric(opted_out.marker)

    _assert_refused(opted_out.http.post(f"{EVALUATOR_API}/evaluate/live", json=_live_body(metric)), opted_out)


def test_opted_out_job_stored_with_cloudpickle_metric_stays_readable(opted_out: _OptedOut) -> None:
    got = opted_out.http.get(f"{EVALUATOR_API}/evaluate/jobs/{opted_out.stored_job}")
    listed = opted_out.http.get(f"{EVALUATOR_API}/evaluate/jobs")

    assert got.status_code == 200, got.text
    assert got.json()["spec"]["metrics"][0]["payload"]["kind"] == "cloudpickle"
    assert listed.status_code == 200, listed.text
    assert opted_out.stored_job in [job["name"] for job in listed.json()["data"]]
    assert _unpickle_count(opted_out.marker) == 0


@pytest.mark.timeout(600)
def test_opted_out_builtin_metrics_still_run(opted_out: _OptedOut) -> None:
    live = opted_out.http.post(f"{EVALUATOR_API}/evaluate/live", json=_live_body(_inline_exact_match()))
    _, job = _run_to_completion(opted_out.base_url, opted_out.http, EvaluateJob, _evaluate_spec(_inline_exact_match()))

    assert live.status_code == 200, live.text
    assert job.status == "completed", job.status_details
