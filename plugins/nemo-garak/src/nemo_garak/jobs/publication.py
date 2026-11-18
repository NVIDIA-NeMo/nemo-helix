# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a finished audit to Intake, so it shows up in Experiments.

Reads garak's ``report.jsonl`` and writes one ATIF session per scored model output under an
Evaluation that already exists. Scores travel in ``extra.verifier_result.rewards``, which Intake
turns into evaluator results at ingest, so a session costs one request.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from httpx import AsyncClient
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.tls import client_verify_from_env
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import AtifCreateParams, AtifCreateRequest, AtifStepParam
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_CONCURRENCY = 8
_ATIF_SOURCE_BY_ROLE = {"user": "user", "assistant": "agent", "system": "system"}


class IntakePublicationSpec(BaseModel):
    """Publish the audit's results to Intake, under an Evaluation that already exists."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(
        min_length=1,
        description="Name of the existing Evaluation to publish under. The audit does not create it.",
    )
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        description="Agent name recorded on each published session. Defaults to the audit target's name.",
    )
    required: bool = Field(
        default=True,
        description="Fail the audit when publication fails. The reports are saved first either way.",
    )


class PublicationSpec(BaseModel):
    """Where a completed audit publishes its results, beyond its own report artifacts."""

    model_config = ConfigDict(extra="forbid")

    intake: IntakePublicationSpec | None = Field(
        default=None, description="Publish results to Intake. Omit to publish nowhere."
    )


class PublicationOutcome(BaseModel):
    """What publication did, as reported in the audit output."""

    status: HelixJobStatus = Field(description="COMPLETED when everything published, ERROR otherwise.")
    evaluation_id: str
    session_count: int = Field(default=0, description="Sessions published (partial on failure).")
    unscored_count: int = Field(default=0, description="Outputs left out because the detector did not score them.")
    error: str | None = None


class PublicationFailedError(RuntimeError):
    """Publication failed and the spec marked it required."""

    def __init__(self, outcome: PublicationOutcome) -> None:
        super().__init__(outcome.error or "publication failed")
        self.outcome = outcome


def _text(message: object) -> str:
    text = message.get("text") if isinstance(message, dict) else message
    return text if isinstance(text, str) else ""


def report_sessions(
    report_path: Path, *, evaluation_id: str, agent_name: str, model_name: str, eval_threshold: float
) -> tuple[list[AtifCreateParams], int]:
    """One ATIF session per scored output in a garak report, and how many outputs went unscored.

    An attempt holds one output per generation and one score per output per detector. A score of
    ``None`` means the detector did not judge that output; garak leaves those out of its pass rate.
    An output passes when its score is below ``eval_threshold``, as in garak's own evaluator.

    Session ids and timestamps come from the report alone, so publishing it again replaces the
    sessions. The start time is the lead run's (``orig_start_time`` on an aggregated report).
    """
    with report_path.open(encoding="utf-8") as report:
        entries = [json.loads(line) for line in report if line.strip()]
    timestamp: str | None = None
    sessions: list[AtifCreateParams] = []
    unscored = 0
    for entry in entries:
        if entry["entry_type"] == "init":
            # garak writes naive local time. It is read as UTC so the timestamp is the same
            # whichever host publishes the report; Intake keys a session's rows on it.
            started = datetime.fromisoformat(entry.get("orig_start_time") or entry["start_time"])
            timestamp = (started if started.tzinfo else started.replace(tzinfo=UTC)).isoformat()
        # garak also logs each attempt before detection runs (status 1); those carry no scores.
        if entry["entry_type"] != "attempt" or entry["status"] != 2:
            continue
        if timestamp is None:
            raise ValueError(f"garak report {report_path} has no 'init' entry before its attempts.")
        probe = entry["probe_classname"]
        prompt: list[AtifStepParam] = [
            {"source": _ATIF_SOURCE_BY_ROLE.get(turn["role"], "user"), "message": _text(turn["content"])}
            for turn in entry["prompt"]["turns"]
        ]
        for detector, scores in entry["detector_results"].items():
            for index, score in enumerate(scores):
                if score is None:
                    unscored += 1
                    continue
                steps = [*prompt, {"source": "agent", "message": _text(entry["outputs"][index])}]
                sessions.append(
                    {
                        "schema_version": "ATIF-v1.7",
                        # The evaluation is part of the id so one audit can be published under two.
                        "session_id": f"garak:{evaluation_id}:{entry['uuid']}:{detector}:{index}",
                        "agent": {"name": agent_name, "version": "unknown", "model_name": model_name},
                        "evaluation_context": {
                            "evaluation_name": evaluation_id,
                            "test_case_name": f"garak/{probe}/{detector}",
                        },
                        # Every step carries the run's start time: Intake keys a session's rows on
                        # it, so a step left to the ingest clock would duplicate on re-publish.
                        "steps": [
                            {**step, "step_id": step_id, "timestamp": timestamp}
                            for step_id, step in enumerate(steps, start=1)
                        ],
                        "extra": {
                            "task_id": f"garak/{probe}",
                            "verifier_result": {
                                "rewards": {
                                    "pass_rate": 1.0 if score < eval_threshold else 0.0,
                                    "detector_score": float(score),
                                }
                            },
                        },
                    }
                )
    return sessions, unscored


async def _publish(
    sessions: list[AtifCreateParams], *, intake: AsyncIntakeClient, evaluation_id: str, workspace: str
) -> list[BaseException]:
    """Send every session and return the failures. One failed session does not stop the others."""
    # ATIF ingest refuses an unknown Evaluation once per session, so check once up front.
    await intake.get_evaluation(name=evaluation_id, workspace=workspace)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def _send(body: AtifCreateParams) -> None:
        async with semaphore:
            await intake.create_atif(workspace=workspace, body=AtifCreateRequest(root=body))

    results = await asyncio.gather(*(_send(body) for body in sessions), return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


async def _with_intake(
    client: NemoClient | None,
    async_client: AsyncNemoClient | None,
    use: Callable[[AsyncIntakeClient], Awaitable[T]],
) -> T:
    """Call ``use`` with an Intake client that carries the platform client's identity, on this loop.

    The sync client is converted. The async client is bound to its caller's loop, so it is cloned
    onto a throwaway transport that keeps its request hooks and CA bundle.
    """
    if client is not None:
        async with client.to_async() as converted:
            return await use(AsyncIntakeClient.from_client(converted))
    assert async_client is not None
    hooks = {event: list(handlers) for event, handlers in async_client._client.event_hooks.items()}
    async with AsyncClient(
        timeout=async_client._client.timeout, event_hooks=hooks, verify=client_verify_from_env()
    ) as http_client:
        return await use(AsyncIntakeClient.from_client(async_client.with_http_client(http_client)))


def publish_audit_report(
    report_path: Path,
    *,
    spec: IntakePublicationSpec,
    workspace: str,
    target_name: str,
    model_name: str,
    eval_threshold: float,
    client: NemoClient | None = None,
    async_client: AsyncNemoClient | None = None,
) -> PublicationOutcome:
    """Publish the garak report at ``report_path`` to Intake and describe what happened.

    Raises :class:`PublicationFailedError` when publication fails and ``spec.required`` is set, and
    returns a failed outcome otherwise. Call it again with a saved ``report.jsonl`` to re-publish.
    """
    outcome = PublicationOutcome(status=HelixJobStatus.COMPLETED, evaluation_id=spec.evaluation_id)
    try:
        if client is None and async_client is None:
            raise RuntimeError("No platform client available; cannot publish to Intake.")
        sessions, outcome.unscored_count = report_sessions(
            report_path,
            evaluation_id=spec.evaluation_id,
            agent_name=spec.agent_name or target_name,
            model_name=model_name,
            eval_threshold=eval_threshold,
        )
        if not sessions:
            raise ValueError("The garak report has no scored results to publish.")

        def _run() -> list[BaseException]:
            return asyncio.run(
                _with_intake(
                    client,
                    async_client,
                    lambda intake: _publish(
                        sessions, intake=intake, evaluation_id=spec.evaluation_id, workspace=workspace
                    ),
                )
            )

        # On its own thread, because the caller may already be inside an event loop (a notebook is).
        with ThreadPoolExecutor(max_workers=1) as pool:
            failures = pool.submit(_run).result()
        outcome.session_count = len(sessions) - len(failures)
        if failures:
            raise RuntimeError(f"{len(failures)} of {len(sessions)} session(s) failed; first: {failures[0]!r}")
    except NotFoundError:
        outcome.error = (
            f"Evaluation {spec.evaluation_id!r} does not exist in workspace {workspace!r}. "
            "Create it before running the audit; the audit does not create it."
        )
    except Exception as error:
        outcome.error = f"{type(error).__name__}: {error}"
    if outcome.error is None:
        return outcome
    outcome.status = HelixJobStatus.ERROR
    if spec.required:
        raise PublicationFailedError(outcome)
    logger.warning("Publication to Intake failed but was not required: %s", outcome.error)
    return outcome
