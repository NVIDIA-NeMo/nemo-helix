# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator plugin job resources for status polling and result downloads."""

from __future__ import annotations

import asyncio
import json
import logging
import tarfile
import time
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import TypeAlias

from nemo_evaluator.jobs.agent_spec import AgentEvalSpec
from nemo_evaluator.jobs.evaluate import EvaluateSpec
from nemo_evaluator.sdk.utils import filter_aggregate_scores
from nemo_evaluator_sdk.execution.job_poll import async_poll_until_terminal
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, AggregateFieldName, EvaluationResult, RowScore
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.jobs.api_factory import BaseJob
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nemo_helix_plugin.jobs.schemas import HelixJobStatusResponse
from pydantic import BaseModel

EvaluatorJob: TypeAlias = BaseJob[EvaluateSpec]
#: The agent counterpart. Separate because the two jobs' specs genuinely differ — an agent job
#: carries tasks and a runner target where a row job carries metrics and a dataset — so reusing
#: `EvaluatorJob` here would fail validation on every field of the spec that came back.
AgentEvaluatorJob: TypeAlias = BaseJob[AgentEvalSpec]

_TERMINAL_FAILURE_STATUSES = frozenset({"error", "cancelled", "failed"})
_TERMINAL_SUCCESS_STATUS = "completed"
_TERMINAL_STATUSES = _TERMINAL_FAILURE_STATUSES | frozenset({_TERMINAL_SUCCESS_STATUS})
_QUEUED_STATUSES = frozenset({"created", "pending"})
_RUNNING_STATUSES = frozenset({"active", "cancelling", "paused", "pausing", "resuming"})
_DEFAULT_POLL_INTERVAL_SECONDS = 10.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 3600.0
_DEFAULT_PENDING_TIMEOUT_SECONDS = 600.0

_RowScorePayload: TypeAlias = RowScore | BaseModel | Mapping[str, object]
_AggregateScoresPayload: TypeAlias = AggregatedMetricResult | BaseModel | Mapping[str, object]

log = logging.getLogger(__name__)


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def metric_job_status_value(status: HelixJobStatusResponse) -> str:
    """Return the normalized status value from a metric job status response."""
    raw = status.status
    return raw.lower() if isinstance(raw, str) else ""


def metric_job_status_details_value(status: HelixJobStatusResponse) -> Mapping[str, object] | None:
    """Return status details from a metric job status response."""
    return status.status_details or None


def _coerce_row_score(row_score: _RowScorePayload) -> RowScore:
    """Convert a platform SDK row-score object to the evaluator SDK value type."""
    if isinstance(row_score, RowScore):
        return row_score
    if isinstance(row_score, BaseModel):
        return RowScore.model_validate(row_score.model_dump(mode="json"))
    return RowScore.model_validate(row_score)


def _coerce_aggregate_scores(aggregate_scores: _AggregateScoresPayload) -> AggregatedMetricResult:
    """Convert platform SDK aggregate scores to the evaluator SDK value type."""
    if isinstance(aggregate_scores, AggregatedMetricResult):
        return aggregate_scores
    if isinstance(aggregate_scores, BaseModel):
        return AggregatedMetricResult.model_validate(aggregate_scores.model_dump(mode="json"))
    return AggregatedMetricResult.model_validate(aggregate_scores)


def _parse_row_scores_jsonl(payload: str) -> list[RowScore]:
    """Parse row-score JSONL downloaded from the evaluator plugin result route."""
    row_scores: list[RowScore] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if stripped:
            row_scores.append(RowScore.model_validate_json(stripped))
    return row_scores


def _parse_aggregate_scores_json(payload: bytes) -> AggregatedMetricResult:
    """Parse aggregate-score JSON downloaded from the evaluator plugin result route."""
    return AggregatedMetricResult.model_validate_json(payload)


def _extract_artifacts_tarball(payload: bytes, output_path: Path) -> Path:
    """Extract a job artifacts tarball into ``output_path`` and return it.

    The evaluator artifact route returns a tarball produced by the Jobs result
    serializer. Validate every member before extraction so a malformed archive
    cannot write outside the selected destination or create links/special files.
    """
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as tar:
        safe_extract_tar(tar, output_path, error_cls=ValueError)
    return output_path


def _artifact_output_path(path: Path | str | None, job_name: str) -> Path:
    """Return the job-specific local directory for downloaded artifacts.

    Validates ``job_name`` so it cannot escape the chosen base directory. Job
    names flow in from API responses and could otherwise contain path
    separators or ``..`` segments that would land the extracted artifacts
    outside the intended location.
    """
    job_path = Path(job_name)
    if job_path.is_absolute() or len(job_path.parts) != 1 or job_name in {".", ".."}:
        raise ValueError(f"Invalid job name for artifact path: {job_name!r}")
    base = Path(".") if path is None else Path(path)
    return base / job_name


def _raise_for_terminal_status(status: HelixJobStatusResponse) -> None:
    """Raise when a terminal evaluator job status is not successful."""
    status_value = metric_job_status_value(status)
    if status_value == _TERMINAL_SUCCESS_STATUS:
        return
    if status_value in _TERMINAL_FAILURE_STATUSES:
        error_details = status.error_details or "Unknown error"
        raise RuntimeError(
            f"NeMo Helix metric job {status.name!r} finished with status {status_value!r}: {error_details}"
        )
    raise RuntimeError(f"NeMo Helix metric job {status.name!r} reached unexpected terminal status {status_value!r}")


def _status_is_complete(status: HelixJobStatusResponse, raise_if_not_complete: bool) -> bool:
    """Return whether the job completed and optionally raise for other statuses."""
    status_value = metric_job_status_value(status)
    if status_value == _TERMINAL_SUCCESS_STATUS:
        return True
    if status_value in _TERMINAL_FAILURE_STATUSES:
        msg = f"NeMo Helix metric job {status.name!r} stopped with status {status_value!r}"
        if raise_if_not_complete:
            error_details = status.error_details or "Unknown error"
            raise RuntimeError(f"{msg}: {error_details}")
        return False
    if status_value in _QUEUED_STATUSES | _RUNNING_STATUSES:
        if raise_if_not_complete:
            raise RuntimeError(f"NeMo Helix metric job {status.name!r} is not complete; status is {status_value!r}")
        return False
    msg = f"NeMo Helix metric job {status.name!r} is in an unknown state: {status_value!r}"
    if raise_if_not_complete:
        raise RuntimeError(msg)
    log.error(msg)
    return False


def _poll_until_terminal(
    get_status: Callable[[], HelixJobStatusResponse],
    *,
    job_name: str,
    timeout: float,
    pending_timeout: float,
    poll_interval: float,
) -> HelixJobStatusResponse:
    """Synchronously poll an evaluator job until it reaches a terminal status."""
    elapsed = 0.0
    pending_elapsed = 0.0

    while True:
        poll_start = monotonic()
        status_response = get_status()
        status = metric_job_status_value(status_response)
        log_payload: dict[str, object] = {
            "job_name": job_name,
            "status": status or "<empty>",
            "elapsed_s": round(elapsed, 1),
            "poll_interval_s": poll_interval,
        }
        if status == "pending":
            log_payload["pending_elapsed_s"] = round(pending_elapsed, 1)
        details = metric_job_status_details_value(status_response)
        if details:
            log_payload["status_details"] = dict(details)
        log.info(json.dumps(log_payload, separators=(",", ":")))

        if status in _TERMINAL_STATUSES:
            return status_response

        poll_duration = monotonic() - poll_start
        if status == "pending":
            pending_elapsed += poll_duration
            if pending_elapsed >= pending_timeout:
                raise TimeoutError(f"'{job_name}' stuck in pending after {pending_timeout}s.")
        else:
            elapsed += poll_duration
            if elapsed >= timeout:
                raise TimeoutError(f"'{job_name}' timed out after {timeout}s. Status: {status}")

        sleep_start = monotonic()
        _pause(poll_interval)
        sleep_duration = monotonic() - sleep_start
        if status == "pending":
            pending_elapsed += sleep_duration
        else:
            elapsed += sleep_duration


class AgentEvaluatorJobResource:
    """High-level SDK handle for a submitted agent-evaluation job.

    Deliberately *not* a subclass of, or base for, :class:`EvaluatorJobResource`. The polling half
    of the two is identical today, but the result half is not: a row evaluation publishes
    ``aggregate-scores`` and ``row-scores`` artifacts, while an agent evaluation publishes
    ``agent-eval-results`` and ``summary`` and produces trials rather than rows. Sharing a base to
    save the polling duplication would put :meth:`EvaluatorJobResource.get_result` on this class,
    where it would fetch row-score routes an agent job never writes — a method whose type says
    "results" and whose behaviour is a 404. The duplication is the cheaper mistake while the two
    shapes are still settling.

    **No ``get_result`` yet.** Reading the trial bundle back means downloading and parsing this
    job's artifacts, which nothing does today, and the name ``AgentEvalResult`` is already taken by
    two different types — the persisted API record and the SDK's trials-and-scores bundle. That
    distinction deserves its own change rather than being settled as a side effect of submission.
    Until then, the persisted record is queryable::

        evaluator.agent_eval_results.list(job_id=job.name)

    **No ``download_artifacts`` either**, for the same reason: the row job publishes an ``artifacts``
    result and this one does not — it saves ``agent-eval-results`` and ``summary``. Offering the
    row job's downloader here would 404. What remains is status and polling, which the platform
    serves per job regardless of what produced it.
    """

    def __init__(
        self,
        *,
        job: AgentEvaluatorJob,
        client: EvaluatorClient,
        workspace: str,
    ) -> None:
        """Store the job identity and typed client used for status calls."""
        self._job = job
        self._client = client
        self._workspace = workspace

    @property
    def name(self) -> str:
        """Return the agent-evaluation job name."""
        return self._job.name

    @property
    def job(self) -> AgentEvaluatorJob:
        """Return the raw job payload captured at resource creation."""
        return self._job

    def get_job_status(self) -> HelixJobStatusResponse:
        """Fetch the current job status from the evaluator plugin API."""
        return self._client.get_agent_eval_job_status(workspace=self._workspace, name=self.name).data()

    def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the job has completed.

        Args:
            raise_if_not_complete: When true, raise ``RuntimeError`` for any status other than
                ``completed``.

        Returns:
            ``True`` only when the current status is ``completed``.
        """
        return _status_is_complete(self.get_job_status(), raise_if_not_complete)

    def wait_until_done(
        self,
        *,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Block until the job reaches a terminal platform status.

        Raises:
            RuntimeError: If the job reaches a terminal failure status.
            TimeoutError: If polling exceeds configured timeouts.
        """
        status = _poll_until_terminal(
            self.get_job_status,
            job_name=self.name,
            timeout=job_timeout_seconds,
            pending_timeout=pending_timeout_seconds,
            poll_interval=poll_interval_seconds,
        )
        _raise_for_terminal_status(status)


class EvaluatorJobResource:
    """High-level SDK handle for a submitted evaluator plugin job."""

    def __init__(
        self,
        *,
        job: EvaluatorJob,
        client: EvaluatorClient,
        workspace: str,
    ) -> None:
        """Store the job identity and typed client used for status and result calls."""
        self._job = job
        self._client = client
        self._workspace = workspace

    @property
    def name(self) -> str:
        """Return the evaluator job name."""
        return self._job.name

    @property
    def job(self) -> EvaluatorJob:
        """Return the raw evaluator job payload captured at resource creation."""
        return self._job

    def get_job_status(self) -> HelixJobStatusResponse:
        """Fetch the current evaluator job status from the evaluator plugin API."""
        return self._client.get_evaluate_job_status(workspace=self._workspace, name=self.name).data()

    def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the evaluator job has completed.

        Args:
            raise_if_not_complete: When true, raise ``RuntimeError`` for any
                status other than ``completed``.

        Returns:
            ``True`` only when the current status is ``completed``.
        """
        return _status_is_complete(self.get_job_status(), raise_if_not_complete)

    def wait_until_done(
        self,
        *,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Block until the evaluator job reaches a terminal platform status.

        Raises:
            RuntimeError: If the job reaches a terminal failure status.
            TimeoutError: If polling exceeds configured timeouts.
        """
        status = _poll_until_terminal(
            self.get_job_status,
            job_name=self.name,
            timeout=job_timeout_seconds,
            pending_timeout=pending_timeout_seconds,
            poll_interval=poll_interval_seconds,
        )
        _raise_for_terminal_status(status)

    def get_result(self, aggregate_fields: tuple[AggregateFieldName, ...] | None = None) -> EvaluationResult:
        """Get aggregate and row-score artifacts as an ``EvaluationResult``."""
        aggregate_payload = self._client.download_evaluate_job_aggregate_scores(
            workspace=self._workspace,
            name=self.name,
        ).read()
        row_scores_payload = self._client.download_evaluate_job_row_scores(
            workspace=self._workspace,
            name=self.name,
        ).read()
        aggregate_scores = filter_aggregate_scores(
            _parse_aggregate_scores_json(aggregate_payload),
            aggregate_fields,
        )
        return EvaluationResult(
            row_scores=_parse_row_scores_jsonl(row_scores_payload.decode("utf-8")),
            aggregate_scores=aggregate_scores,
        )

    def download_artifacts(self, path: Path | str | None = None) -> Path:
        """Download and extract the full evaluator job artifacts tarball.

        Args:
            path: Base output directory. Defaults to the current directory.

        Returns:
            The job-specific directory that contains the extracted artifacts.
        """
        payload = self._client.download_evaluate_job_artifacts(workspace=self._workspace, name=self.name).read()
        return _extract_artifacts_tarball(
            payload,
            _artifact_output_path(path, self.name),
        )


class AsyncAgentEvaluatorJobResource:
    """Async high-level SDK handle for a submitted agent-evaluation job.

    The async counterpart of :class:`AgentEvaluatorJobResource`, with the same surface: status and
    polling only. An agent evaluation publishes ``agent-eval-results`` and ``summary`` rather than
    the row job's ``aggregate-scores`` and ``artifacts``, so ``get_result`` and
    ``download_artifacts`` are omitted — on this job they would 404. Read the persisted record
    instead::

        await evaluator.agent_eval_results.list(job_id=job.name)
    """

    def __init__(
        self,
        *,
        job: AgentEvaluatorJob,
        client: AsyncEvaluatorClient,
        workspace: str,
    ) -> None:
        """Store the job identity and typed async client used for status calls."""
        self._job = job
        self._client = client
        self._workspace = workspace

    @property
    def name(self) -> str:
        """Return the agent-evaluation job name."""
        return self._job.name

    @property
    def job(self) -> AgentEvaluatorJob:
        """Return the raw job payload captured at resource creation."""
        return self._job

    async def get_job_status(self) -> HelixJobStatusResponse:
        """Fetch the current job status from the evaluator plugin API."""
        response = await self._client.get_agent_eval_job_status(workspace=self._workspace, name=self.name)
        return response.data()

    async def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the job has completed.

        Args:
            raise_if_not_complete: When true, raise ``RuntimeError`` for any status other than
                ``completed``.

        Returns:
            ``True`` only when the current status is ``completed``.
        """
        return _status_is_complete(await self.get_job_status(), raise_if_not_complete)

    async def wait_until_done(
        self,
        *,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Wait until the job reaches a terminal platform status.

        Raises:
            RuntimeError: If the job reaches a terminal failure status.
            TimeoutError: If polling exceeds configured timeouts.
        """
        status = await async_poll_until_terminal(
            self.get_job_status,
            status_value=metric_job_status_value,
            details_value=metric_job_status_details_value,
            job_name=self.name,
            terminal=_TERMINAL_STATUSES,
            timeout=job_timeout_seconds,
            pending_timeout=pending_timeout_seconds,
            poll_interval=poll_interval_seconds,
        )
        _raise_for_terminal_status(status)


class AsyncEvaluatorJobResource:
    """Async high-level SDK handle for a submitted evaluator plugin job."""

    def __init__(
        self,
        *,
        job: EvaluatorJob,
        client: AsyncEvaluatorClient,
        workspace: str,
    ) -> None:
        """Store the job identity and typed client used for async status and result calls."""
        self._job = job
        self._client = client
        self._workspace = workspace

    @property
    def name(self) -> str:
        """Return the evaluator job name."""
        return self._job.name

    @property
    def job(self) -> EvaluatorJob:
        """Return the raw evaluator job payload captured at resource creation."""
        return self._job

    async def get_job_status(self) -> HelixJobStatusResponse:
        """Fetch the current evaluator job status from the evaluator plugin API."""
        response = await self._client.get_evaluate_job_status(workspace=self._workspace, name=self.name)
        return response.data()

    async def check_if_complete(self, *, raise_if_not_complete: bool = False) -> bool:
        """Return whether the evaluator job has completed.

        Args:
            raise_if_not_complete: When true, raise ``RuntimeError`` for any
                status other than ``completed``.

        Returns:
            ``True`` only when the current status is ``completed``.
        """
        return _status_is_complete(await self.get_job_status(), raise_if_not_complete)

    async def wait_until_done(
        self,
        *,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        job_timeout_seconds: float = _DEFAULT_JOB_TIMEOUT_SECONDS,
        pending_timeout_seconds: float = _DEFAULT_PENDING_TIMEOUT_SECONDS,
    ) -> None:
        """Wait until the evaluator job reaches a terminal platform status.

        Raises:
            RuntimeError: If the job reaches a terminal failure status.
            TimeoutError: If polling exceeds configured timeouts.
        """
        status = await async_poll_until_terminal(
            self.get_job_status,
            status_value=metric_job_status_value,
            details_value=metric_job_status_details_value,
            job_name=self.name,
            terminal=_TERMINAL_STATUSES,
            timeout=job_timeout_seconds,
            pending_timeout=pending_timeout_seconds,
            poll_interval=poll_interval_seconds,
        )
        _raise_for_terminal_status(status)

    async def get_result(
        self,
        aggregate_fields: tuple[AggregateFieldName, ...] | None = None,
    ) -> EvaluationResult:
        """Get aggregate and row-score artifacts as an ``EvaluationResult``.

        Aggregate and row-score downloads are dispatched concurrently so a slow
        artifact never serializes the other.
        """
        aggregate_payload, row_scores_payload = await asyncio.gather(
            self._read_aggregate_scores(),
            self._read_row_scores(),
        )
        aggregate_scores = filter_aggregate_scores(
            _parse_aggregate_scores_json(aggregate_payload),
            aggregate_fields,
        )
        return EvaluationResult(
            row_scores=_parse_row_scores_jsonl(row_scores_payload.decode("utf-8")),
            aggregate_scores=aggregate_scores,
        )

    async def download_artifacts(self, path: Path | str | None = None) -> Path:
        """Download and extract the full evaluator job artifacts tarball.

        Args:
            path: Base output directory. Defaults to the current directory.

        Returns:
            The job-specific directory that contains the extracted artifacts.
        """
        payload = await self._read_artifacts()
        return await asyncio.to_thread(
            _extract_artifacts_tarball,
            payload,
            _artifact_output_path(path, self.name),
        )

    async def _read_aggregate_scores(self) -> bytes:
        """Read aggregate scores through the async typed client."""
        response = await self._client.download_evaluate_job_aggregate_scores(
            workspace=self._workspace,
            name=self.name,
        )
        return await response.read()

    async def _read_row_scores(self) -> bytes:
        """Read row scores through the async typed client."""
        response = await self._client.download_evaluate_job_row_scores(
            workspace=self._workspace,
            name=self.name,
        )
        return await response.read()

    async def _read_artifacts(self) -> bytes:
        """Read job artifacts through the async typed client."""
        response = await self._client.download_evaluate_job_artifacts(workspace=self._workspace, name=self.name)
        return await response.read()
