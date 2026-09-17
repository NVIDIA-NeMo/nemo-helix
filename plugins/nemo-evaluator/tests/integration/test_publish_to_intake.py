# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: publish_to_intake against a live Intake + ClickHouse.

Marked ``integration`` (auto-applied to ``/integration/`` paths), so it runs under
``make test-integration`` / ``-m integration`` and is excluded from the unit suite.
Session fixtures stand up ClickHouse (Docker) and the platform
(``auth,entities,intake``); the test skips cleanly when Docker is unavailable.

Run directly::

    uv run pytest plugins/nemo-evaluator/tests/integration/test_publish_to_intake.py -v

Requires Docker (Intake is ClickHouse-backed) and a free :8123. The platform binds the port from
``NMP_BASE_URL`` (default :8080), so set it to run alongside a local dev platform::

    NMP_BASE_URL=http://localhost:8096 uv run pytest ...
"""

from __future__ import annotations

import math
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from nemo_evaluator.intake.publish import PublishReport, publish_to_intake
from nemo_evaluator.intake.row_adapter import row_result_to_agent_eval_result
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, EvaluationResult, RowScore
from nemo_platform.types.intake.trace_filter_param import TraceFilterParam
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.sdk import AsyncNeMoPlatform

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_URL = os.environ.get("NMP_BASE_URL", "http://localhost:8080")
WORKSPACE = "default"
GROUP_NAME = "intake-it-group"
EXPERIMENT_NAME = "intake-it-exp"
RUN_ID = "intake-it-run"
NAN_EXPERIMENT_NAME = "intake-it-nan-exp"
NAN_RUN_ID = "intake-it-nan-run"
IDEMPOTENCY_EXPERIMENT_NAME = "intake-it-idempotency-exp"
IDEMPOTENCY_RUN_ID = "intake-it-idempotency-run"
OTLP_EXPERIMENT_NAME = "intake-it-otlp-exp"
OTLP_RUN_ID = "intake-it-otlp-run"
OTLP_TRACE_ID = "0123456789abcdef0123456789abcdef"
OTLP_ROOT_SPAN_ID = "0102030405060708"
ROW_EXPERIMENT_NAME = "intake-it-row-exp"
ROW_RUN_ID = "intake-it-row-run"
#: Recent, not fixed: a trajectory's start_time is a real timestamp, and Intake's trace queries
#: only look back a bounded window — a hardcoded past date ingests fine but reads back empty.
STARTED_AT = datetime.now(UTC) - timedelta(minutes=5)
#: Fixed for the process so a re-publish writes the same ReplacingMergeTree key.
_OTLP_START_NANOS = time.time_ns()


def _docker_available() -> bool:
    if find_spec("docker") is None:
        return False
    from docker.errors import DockerException

    import docker

    try:
        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
        return True
    except (DockerException, OSError):
        return False


def _wait_for_tcp(host: str, port: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
    raise RuntimeError(f"{host}:{port} not reachable within {timeout}s")


def _wait_for_ready(base_url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health/ready", timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"platform at {base_url} not ready within {timeout}s")


#: Stable, not per-session: `--remove` only matches containers whose data directory equals the one
#: being asked for, so a per-session path leaves a stranded container unremovable.
_CLICKHOUSE_DATA_DIR = REPO_ROOT / "tmp" / "evaluator-intake-clickhouse"


#: Bounds one provisioner call, so a wedged Docker fails this fixture instead of the worker.
_CLICKHOUSE_SCRIPT_TIMEOUT_ENV = "NMP_EVALUATOR_CLICKHOUSE_SCRIPT_TIMEOUT"


def _script_timeout_seconds() -> float:
    """Resolve the provisioner timeout. Zero and negative time out immediately; ``inf`` never."""
    raw = os.getenv(_CLICKHOUSE_SCRIPT_TIMEOUT_ENV)
    if raw is None:
        return 300.0
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError(f"{_CLICKHOUSE_SCRIPT_TIMEOUT_ENV} must be a number of seconds, got {raw!r}") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{_CLICKHOUSE_SCRIPT_TIMEOUT_ENV} must be a positive, finite number of seconds, got {raw!r}")
    return seconds


_CLICKHOUSE_SCRIPT_TIMEOUT_SECONDS = _script_timeout_seconds()


def _run_clickhouse_script(*args: str, env: dict[str, str], check: bool) -> int:
    """Invoke the local ClickHouse provisioner script and return its exit status."""
    return subprocess.run(
        ["bash", str(REPO_ROOT / "services/intake/scripts/spans/run_clickhouse.sh"), *args],
        check=check,
        cwd=REPO_ROOT,
        env=env,
        timeout=_CLICKHOUSE_SCRIPT_TIMEOUT_SECONDS,
    ).returncode


@pytest.fixture(scope="session")
def _clickhouse() -> Iterator[None]:
    if not _docker_available():
        pytest.skip("Docker not available; required for ClickHouse-backed Intake")
    clickhouse_env = {**os.environ, "CLICKHOUSE_DATA_DIR": str(_CLICKHOUSE_DATA_DIR)}
    # Reclaim anything a previous session left behind. Not check=True: usually there is nothing to
    # remove, and a failure here must not mask the provisioning error below.
    try:
        reclaimed = _run_clickhouse_script("--remove", env=clickhouse_env, check=False)
    except subprocess.TimeoutExpired:
        reclaimed = -1
    # The directory is bind-mounted read-write, so only wipe it once removal succeeded — otherwise
    # a still-running ClickHouse loses its data underneath it.
    if reclaimed == 0:
        # Not ignore_errors: a partial delete leaves the session running against state it believes
        # is empty.
        try:
            shutil.rmtree(_CLICKHOUSE_DATA_DIR)
        except FileNotFoundError:
            pass
    try:
        _run_clickhouse_script(env=clickhouse_env, check=True)
        _wait_for_tcp("localhost", 8123, timeout=60)
        yield
    finally:
        # Best effort: setup reclaims on its own, so this must not mask a real test failure.
        try:
            _run_clickhouse_script("--remove", env=clickhouse_env, check=False)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture(scope="session")
def platform_base_url(_clickhouse: None) -> Iterator[str]:
    # Bind the port from BASE_URL rather than letting `services run` fall back to its 8080 default:
    # NMP_BASE_URL is client-side only, so without this the suite silently requires 8080 to be free
    # and cannot run alongside a local dev platform. Mirrors the sibling fixtures in conftest, which
    # each take their own port for the same reason.
    port = urlsplit(BASE_URL).port or 8080
    process = subprocess.Popen(
        ["uv", "run", "nemo", "services", "run", "--services", "auth,entities,intake", "--port", str(port)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "NMP_BASE_URL": BASE_URL,
            "NMP_INTAKE_CLICKHOUSE_URL": "http://localhost:8123",
        },
    )
    try:
        _wait_for_ready(BASE_URL, timeout=180)
        yield BASE_URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def _result() -> AgentEvalResult:
    trials = [
        AgentEvalTrial(
            id="trial-1",
            task_id="task-1",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text="The capital of France is Paris."),
        ),
        AgentEvalTrial(
            id="trial-2",
            task_id="task-2",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text="2 + 2 = 4."),
        ),
    ]
    scores = [
        AgentEvalTaskScore(
            id="score-1",
            run_id=RUN_ID,
            task_id="task-1",
            trial_id="trial-1",
            metric_type="accuracy",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="score", value=1.0), MetricOutput(name="passed", value=True)],
        ),
        AgentEvalTaskScore(
            id="score-2",
            run_id=RUN_ID,
            task_id="task-1",
            trial_id="trial-1",
            metric_type="judge",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="verdict", value="correct")],
        ),
        AgentEvalTaskScore(
            id="score-3",
            run_id=RUN_ID,
            task_id="task-2",
            trial_id="trial-2",
            metric_type="accuracy",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="score", value=0.0), MetricOutput(name="passed", value=False)],
        ),
    ]
    return AgentEvalResult(
        run_id=RUN_ID,
        tasks=[],
        trials=trials,
        scores=scores,
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


async def test_publish_to_intake_round_trip(platform_base_url: str) -> None:
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        # Precondition: the Experiment must exist before ingest.
        group = await async_sdk.experiments.create(
            workspace=WORKSPACE, name=GROUP_NAME, description="Intake IT", exist_ok=True
        )
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        report = await publish_to_intake(
            _result(),
            client=client_from_platform(async_sdk, AsyncIntakeClient),
            experiment_id=EXPERIMENT_NAME,
            workspace=WORKSPACE,
            agent_name="intake-it-agent",
            model_name="intake-it-model",
        )

        assert report.trial_count == 2
        assert report.evaluator_result_count == 5
        published = {trial.trial_id: trial for trial in report.published_trials}

        # --- trial-1: trajectory + experiment-context propagation, read back via the Intake API.
        t1 = published["trial-1"]
        trace_filter: TraceFilterParam = {"session_id": t1.session_id}
        traces = [trace async for trace in async_sdk.intake.traces.list(workspace=WORKSPACE, filter=trace_filter)]
        assert len(traces) == 1
        trace = traces[0]
        assert trace.session_id == t1.session_id
        assert trace.root_span_id == t1.span_id
        assert trace.evaluation_context is not None
        evaluation_context = trace.evaluation_context.to_dict()
        assert evaluation_context["evaluation_name"] == EXPERIMENT_NAME
        assert evaluation_context["test_case_name"] == "task-1"

        # --- trial-1 scores: every field, every data_type coercion.
        rows = await async_sdk.intake.spans.evaluator_results.list(t1.span_id, workspace=WORKSPACE)
        by_name = {row.name: row for row in rows}
        assert set(by_name) == {"accuracy.score", "accuracy.passed", "judge.verdict"}
        for row in rows:
            assert row.session_id == t1.session_id
            assert row.span_id == t1.span_id
            assert row.workspace == WORKSPACE
        assert by_name["accuracy.score"].data_type == "NUMERIC"
        assert by_name["accuracy.score"].value == 1.0
        assert by_name["accuracy.passed"].data_type == "BOOLEAN"
        assert by_name["accuracy.passed"].value == 1.0
        assert by_name["judge.verdict"].data_type == "TEXT"
        assert by_name["judge.verdict"].string_value == "correct"

        # --- trial-2: distinct session/span; BOOLEAN false coerces to 0.0.
        t2 = published["trial-2"]
        assert t2.session_id != t1.session_id
        assert t2.span_id != t1.span_id
        rows2 = await async_sdk.intake.spans.evaluator_results.list(t2.span_id, workspace=WORKSPACE)
        by_name2 = {row.name: row for row in rows2}
        assert set(by_name2) == {"accuracy.score", "accuracy.passed"}
        assert by_name2["accuracy.passed"].data_type == "BOOLEAN"
        assert by_name2["accuracy.passed"].value == 0.0
        assert by_name2["accuracy.score"].value == 0.0


def _nan_result() -> AgentEvalResult:
    """A result with a NaN-valued output and a FAILED score alongside one valid score."""
    trial = AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="answer"),
    )
    scores = [
        AgentEvalTaskScore(
            id="score-ok",
            run_id=NAN_RUN_ID,
            task_id="task-1",
            trial_id="trial-1",
            metric_type="accuracy",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="score", value=0.5), MetricOutput(name="broken", value=math.nan)],
        ),
        AgentEvalTaskScore(
            id="score-failed",
            run_id=NAN_RUN_ID,
            task_id="task-1",
            trial_id="trial-1",
            metric_type="judge",
            status=AgentEvalScoreStatus.FAILED,
            outputs=[MetricOutput(name="verdict", value=math.nan)],
        ),
    ]
    return AgentEvalResult(
        run_id=NAN_RUN_ID,
        tasks=[],
        trials=[trial],
        scores=scores,
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


async def test_publish_skips_nan_and_failed_scores(platform_base_url: str) -> None:
    # A NaN value is not representable in JSON and a FAILED score is not a real measurement; neither
    # should reach Intake. Only the finite, completed output should be stored.
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        group = await async_sdk.experiments.create(workspace=WORKSPACE, name=GROUP_NAME, exist_ok=True)
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=NAN_EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-nan-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        report = await publish_to_intake(
            _nan_result(),
            client=client_from_platform(async_sdk, AsyncIntakeClient),
            experiment_id=NAN_EXPERIMENT_NAME,
            workspace=WORKSPACE,
            agent_name="intake-it-agent",
        )

        published = report.published_trials[0]
        rows = await async_sdk.intake.spans.evaluator_results.list(published.span_id, workspace=WORKSPACE)
        assert {row.name for row in rows} == {"accuracy.score"}
        assert report.evaluator_result_count == 1

        # The dropped outputs are surfaced (not silently lost) until Intake can model failure.
        assert {(skip.name, skip.reason) for skip in report.skipped} == {
            ("accuracy.broken", "non-finite value"),
            ("judge.verdict", "scoring failed"),
        }


def _idempotency_result() -> AgentEvalResult:
    """One trial with one score — the smallest result that exercises both write paths."""
    return AgentEvalResult(
        run_id=IDEMPOTENCY_RUN_ID,
        tasks=[],
        trials=[
            AgentEvalTrial(
                id="trial-1",
                task_id="task-1",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="answer"),
            )
        ],
        scores=[
            AgentEvalTaskScore(
                id="score-1",
                run_id=IDEMPOTENCY_RUN_ID,
                task_id="task-1",
                trial_id="trial-1",
                metric_type="accuracy",
                status=AgentEvalScoreStatus.COMPLETED,
                outputs=[MetricOutput(name="score", value=1.0)],
            )
        ],
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


async def test_republishing_the_same_result_is_idempotent(platform_base_url: str) -> None:
    # A job worker can publish successfully and die before recording completion, so a retry must not
    # double-count. Intake's spans table is a ReplacingMergeTree keyed on start_time, which is only
    # stable because the trajectory carries the run's started_at (see mapping.trial_to_atif_ingest);
    # without it each publish lands a second, uncollapsible row per trial.
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        group = await async_sdk.experiments.create(workspace=WORKSPACE, name=GROUP_NAME, exist_ok=True)
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=IDEMPOTENCY_EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-idempotency-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        async def publish() -> PublishReport:
            return await publish_to_intake(
                _idempotency_result(),
                client=client_from_platform(async_sdk, AsyncIntakeClient),
                experiment_id=IDEMPOTENCY_EXPERIMENT_NAME,
                workspace=WORKSPACE,
                agent_name="intake-it-agent",
            )

        first = await publish()
        second = await publish()

        # Same identities both times — nothing is minted per-publish.
        assert first.trial_count == second.trial_count == 1
        assert first.published_trials[0].session_id == second.published_trials[0].session_id
        assert first.published_trials[0].span_id == second.published_trials[0].span_id

        session_id = second.published_trials[0].session_id
        trace_filter: TraceFilterParam = {"session_id": session_id}
        traces = [trace async for trace in async_sdk.intake.traces.list(workspace=WORKSPACE, filter=trace_filter)]
        assert len(traces) == 1, "re-publish duplicated the trajectory instead of replacing it"

        rows = await async_sdk.intake.spans.evaluator_results.list(
            second.published_trials[0].span_id, workspace=WORKSPACE
        )
        assert [row.name for row in rows] == ["accuracy.score"]


def _row_result() -> EvaluationResult:
    """One scored row — the smallest dataset-driven result exercising both write paths."""
    return EvaluationResult(
        row_scores=[
            RowScore(
                row_index=0,
                item={"question": "capital of France?", "qid": "q-1"},
                sample={"output_text": "Paris", "response": {}},
                metrics={"exact_match": [MetricOutput(name="score", value=1.0)]},
                requests=[],
            )
        ],
        aggregate_scores=AggregatedMetricResult(scores=[]),
    )


async def test_row_result_publishes_and_is_idempotent(platform_base_url: str) -> None:
    # The dataset-driven path adapts rows into the publisher's shape rather than using a second
    # mapping, so it inherits the same idempotency guarantee: re-publishing replaces rather than
    # duplicating. Row identity comes from the configured column, not the row's position.
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        group = await async_sdk.experiments.create(workspace=WORKSPACE, name=GROUP_NAME, exist_ok=True)
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=ROW_EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-row-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        async def publish() -> PublishReport:
            adapted = row_result_to_agent_eval_result(
                _row_result(),
                run_id=ROW_RUN_ID,
                started_at=STARTED_AT,
                test_case_id_field="qid",
            )
            return await publish_to_intake(
                adapted,
                client=client_from_platform(async_sdk, AsyncIntakeClient),
                experiment_id=ROW_EXPERIMENT_NAME,
                workspace=WORKSPACE,
                agent_name="intake-it-row-agent",
            )

        first = await publish()
        second = await publish()

        assert first.trial_count == second.trial_count == 1
        session_id = second.published_trials[0].session_id
        assert session_id == f"{ROW_RUN_ID}:q-1"
        assert first.published_trials[0].span_id == second.published_trials[0].span_id

        trace_filter: TraceFilterParam = {"session_id": session_id}
        traces = [trace async for trace in async_sdk.intake.traces.list(workspace=WORKSPACE, filter=trace_filter)]
        assert len(traces) == 1, "re-publish duplicated the row instead of replacing it"
        assert traces[0].evaluation_context is not None
        assert traces[0].evaluation_context.test_case_id == "q-1"

        rows = await async_sdk.intake.spans.evaluator_results.list(
            second.published_trials[0].span_id, workspace=WORKSPACE
        )
        assert [row.name for row in rows] == ["exact_match.score"]


def _otlp_result() -> AgentEvalResult:
    """One trial whose trace is OTLP, so publish takes the OTLP path instead of ATIF.

    Span times are minted once at import and reused by every publish in a run: Intake's
    spans table is a ReplacingMergeTree keyed partly on ``start_time``, so a per-publish
    clock would make a re-publish insert rather than replace. They are recent because that
    table is also TTL'd on ``start_time``, and a fixed past timestamp would be dropped on
    write while ingest still reported success.
    """
    span = {
        "traceId": OTLP_TRACE_ID,
        "spanId": OTLP_ROOT_SPAN_ID,
        "name": "agent run",
        "startTimeUnixNano": str(_OTLP_START_NANOS),
        "endTimeUnixNano": str(_OTLP_START_NANOS + 5_000_000),
        "attributes": [{"key": "output.value", "value": {"stringValue": "answer"}}],
    }
    trace = EvidenceDescriptor(
        kind="trace", format="otlp", data={"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    )
    return AgentEvalResult(
        run_id=OTLP_RUN_ID,
        tasks=[],
        trials=[
            AgentEvalTrial(
                id="trial-1",
                task_id="task-1",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="answer"),
                evidence=CandidateEvidence(descriptors={"trace": trace}),
            )
        ],
        scores=[
            AgentEvalTaskScore(
                id="score-1",
                run_id=OTLP_RUN_ID,
                task_id="task-1",
                trial_id="trial-1",
                metric_type="accuracy",
                status=AgentEvalScoreStatus.COMPLETED,
                outputs=[MetricOutput(name="score", value=1.0)],
            )
        ],
        summary=AgentEvalSummary(),
        metadata=RunMetadata(started_at=STARTED_AT),
    )


async def test_publishing_a_trial_with_an_otlp_trace_lands_its_spans(platform_base_url: str) -> None:
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        group = await async_sdk.experiments.create(workspace=WORKSPACE, name=GROUP_NAME, exist_ok=True)
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=OTLP_EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-otlp-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        report = await publish_to_intake(
            _otlp_result(),
            client=client_from_platform(async_sdk, AsyncIntakeClient),
            experiment_id=OTLP_EXPERIMENT_NAME,
            workspace=WORKSPACE,
            agent_name="intake-it-agent",
        )

        published = report.published_trials[0]
        # The span id is the producer's own, read off the payload rather than queried back.
        assert published.span_id == OTLP_ROOT_SPAN_ID
        assert published.session_id == f"{OTLP_RUN_ID}:trial-1"

        # The stamped session id is what Intake indexed the spans under.
        spans = await async_sdk.intake.spans.list(workspace=WORKSPACE, filter={"session_id": published.session_id})
        assert [span.name for span in spans.data] == ["agent run"]

        assert {span.source for span in spans.data} == {"otel"}

        rows = await async_sdk.intake.spans.evaluator_results.list(published.span_id, workspace=WORKSPACE)
        assert [row.name for row in rows] == ["accuracy.score"]


async def test_republishing_an_otlp_result_replaces_rather_than_duplicates(platform_base_url: str) -> None:
    # The session id we stamp is part of the ReplacingMergeTree key, so getting it wrong or
    # letting it vary per publish inserts a second uncollapsible row instead of replacing.
    async with AsyncNeMoPlatform(base_url=platform_base_url, max_retries=2) as async_sdk:
        group = await async_sdk.experiments.create(workspace=WORKSPACE, name=GROUP_NAME, exist_ok=True)
        await async_sdk.evaluations.create(
            workspace=WORKSPACE,
            name=OTLP_EXPERIMENT_NAME,
            experiment_ids=[group.id],
            dataset_name="intake-it-otlp-dataset",
            dataset_version="v1",
            exist_ok=True,
        )

        async def publish() -> PublishReport:
            return await publish_to_intake(
                _otlp_result(),
                client=client_from_platform(async_sdk, AsyncIntakeClient),
                experiment_id=OTLP_EXPERIMENT_NAME,
                workspace=WORKSPACE,
                agent_name="intake-it-agent",
            )

        first = await publish()
        second = await publish()

        assert first.published_trials[0].session_id == second.published_trials[0].session_id
        assert first.published_trials[0].span_id == second.published_trials[0].span_id

        session_id = second.published_trials[0].session_id
        trace_filter: TraceFilterParam = {"session_id": session_id}
        traces = [trace async for trace in async_sdk.intake.traces.list(workspace=WORKSPACE, filter=trace_filter)]
        assert len(traces) == 1, "re-publish duplicated the OTLP trace instead of replacing it"

        spans = await async_sdk.intake.spans.list(workspace=WORKSPACE, filter={"session_id": session_id})
        assert len(spans.data) == 1, "re-publish duplicated the span instead of replacing it"
