# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a finished audit to Intake, when the spec asked for it.

This module reads garak's ``report.jsonl`` and writes it to Intake under an Evaluation that already
exists, so the audit shows up in Experiments next to other evaluation runs. When the spec has no
``publication.intake`` block nothing here runs. The block has the same fields as the one in the
Helix Evals plugin.

Each published session is one ATIF trajectory. Its scores travel in ``extra.verifier_result.rewards``,
which Intake turns into evaluator results at ingest, so a session costs one request.

Two granularities:

``attempt`` (default)
    One session per scored model output: the prompt garak sent, the output the target returned, and
    the detector's verdict on it. ``pass_rate`` is 1.0 or 0.0, so its mean over a test case is the
    pass rate garak reports for that probe and detector.

``probe``
    One session per probe and detector carrying ``pass_rate`` = passed / evaluated from garak's
    ``eval`` rows. Much smaller, with no prompts or outputs.

Either way the test case is ``garak/<probe>/<detector>``, so the two granularities and audits of
different targets line up in Experiments.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from httpx import AsyncClient
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import AtifAgentParam, AtifCreateParams, AtifCreateRequest, AtifStepParam
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: ATIF schema version the publisher emits.
ATIF_SCHEMA_VERSION = "ATIF-v1.7"

#: Default ``agent.version``. An audit target carries no version and ATIF requires one.
DEFAULT_AGENT_VERSION = "unknown"

#: Ceiling on concurrent session publishes.
DEFAULT_MAX_CONCURRENCY = 8

#: Evaluator names the published sessions carry. ``pass_rate`` is shared by both granularities so
#: an Evaluation published either way aggregates under the same name.
PASS_RATE_REWARD = "pass_rate"
DETECTOR_SCORE_REWARD = "detector_score"

#: How many per-session failures the outcome's error message spells out.
_MAX_REPORTED_FAILURES = 5

PublicationGranularity = Literal["attempt", "probe"]


class IntakePublicationSpec(BaseModel):
    """Publish this audit's results to Intake, under an Evaluation that already exists.

    ``evaluation_id`` is the *name* of a ``client.evaluations`` record, the same value the Helix
    Evals plugin takes in ``publication.intake.evaluation_id``. The audit never creates the
    Evaluation: a missing one is an error.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(
        min_length=1,
        description="Name of the existing Evaluation to publish under. Must already exist; the audit "
        "does not create it.",
    )
    granularity: PublicationGranularity = Field(
        default="attempt",
        description="'attempt' publishes one session per scored model output, with the prompt, the "
        "output and the detector score. 'probe' publishes one session per probe and detector with "
        "the pass rate only; use it when an audit is too large to publish attempt by attempt.",
    )
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        description="Agent name recorded on each published session. Defaults to the audit target's name.",
    )
    agent_version: str = Field(
        default=DEFAULT_AGENT_VERSION,
        min_length=1,
        description="Agent version recorded on each published session. An audit target carries no "
        "version, so this defaults to 'unknown'.",
    )
    required: bool = Field(
        default=True,
        description="Fail the audit when publication fails. Defaults to True so an audit that asked "
        "to publish does not report success with nothing in Experiments. The reports are saved "
        "before publication runs, so a failed audit still leaves them intact to re-publish. Set "
        "False to keep the audit successful and report the failure in its output instead.",
    )


class PublicationSpec(BaseModel):
    """Where a completed audit publishes its results, beyond its own report artifacts."""

    model_config = ConfigDict(extra="forbid")

    intake: IntakePublicationSpec | None = Field(
        default=None, description="Publish results to Intake. Omit to publish nowhere."
    )


class PublicationOutcome(BaseModel):
    """What publication did, as reported in the audit output."""

    model_config = ConfigDict(extra="forbid")

    status: HelixJobStatus = Field(
        description="Platform job status vocabulary: COMPLETED when everything published, ERROR "
        "otherwise. Only those two are ever emitted here.",
    )
    evaluation_id: str = Field(description="Evaluation the results were published under.")
    granularity: PublicationGranularity = Field(description="Granularity the sessions were published at.")
    session_count: int = Field(default=0, description="Sessions actually published (partial on failure).")
    unscored_count: int = Field(
        default=0,
        description="Results left out because garak did not score them: detector results of None in "
        "'attempt' granularity, probe and detector pairs with nothing evaluated in 'probe' granularity.",
    )
    error: str | None = Field(default=None, description="Why publication failed; absent on success.")


class PublicationFailedError(RuntimeError):
    """Publication failed and the spec marked it required.

    Carries the outcome so the caller can still report what landed before the failure.
    """

    def __init__(self, outcome: PublicationOutcome) -> None:
        super().__init__(outcome.error or "publication failed")
        self.outcome = outcome


# --- Reading garak's report ---------------------------------------------------


@dataclass(frozen=True)
class GarakReport:
    """The parts of a garak ``report.jsonl`` that publication reads."""

    run_id: str
    """Stable id of the audit: the lead per-probe run's id on an aggregated report."""
    started_at: datetime
    garak_version: str | None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    evals: list[dict[str, Any]] = field(default_factory=list)


def _parse_start_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # garak writes naive local time. Publication runs where garak ran, so local is the right reading.
    return parsed.astimezone() if parsed.tzinfo is None else parsed


def _iter_entries(report_path: Path) -> Iterator[dict[str, Any]]:
    with report_path.open(encoding="utf-8") as report_fd:
        for line_number, line in enumerate(report_fd, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable line %d of garak report %s", line_number, report_path)
                continue
            if isinstance(entry, dict):
                yield entry


def read_report(report_path: Path) -> GarakReport:
    """Read the attempts, eval rows and run identity out of a garak ``report.jsonl``.

    Only completed attempts (``status == 2``) are kept; garak also logs each attempt once before
    detection runs, and those rows carry no detector results.
    """
    run_id: str | None = None
    started_at: datetime | None = None
    garak_version: str | None = None
    attempts: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    for entry in _iter_entries(report_path):
        entry_type = entry.get("entry_type")
        if entry_type == "init" and run_id is None:
            # An aggregated report restamps `run` and `start_time` every time it is rebuilt and keeps
            # the lead run's originals under `orig_*`. Those are the ones that stay put on re-publish.
            run_id = str(entry.get("orig_uuid") or entry.get("run") or "")
            started_at = _parse_start_time(entry.get("orig_start_time")) or _parse_start_time(entry.get("start_time"))
            version = entry.get("garak_version")
            garak_version = version if isinstance(version, str) else None
        elif entry_type == "attempt" and entry.get("status") == 2:
            attempts.append(entry)
        elif entry_type == "eval":
            evals.append(entry)
    if not run_id:
        raise ValueError(f"garak report {report_path} has no 'init' entry with a run id; cannot publish it.")
    if started_at is None:
        # Session rows are keyed on their start time, so "now" here would make a re-publish duplicate
        # rows. A report with no start time is not one garak wrote.
        raise ValueError(f"garak report {report_path} has no start time in its 'init' entry; cannot publish it.")
    return GarakReport(
        run_id=run_id,
        started_at=started_at,
        garak_version=garak_version,
        attempts=attempts,
        evals=evals,
    )


def _message_text(message: object) -> str:
    """Text of a garak message, across the report shapes garak has used.

    Current garak writes ``{"text": ...}`` objects; older reports wrote bare strings.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        text = message.get("text")
        return text if isinstance(text, str) else ""
    return ""


_ATIF_SOURCE_BY_ROLE = {"user": "user", "assistant": "agent", "system": "system"}


def _prompt_steps(prompt: object) -> list[tuple[str, str]]:
    """``(source, text)`` for each turn of an attempt's prompt.

    Current garak writes a conversation (``{"turns": [{"role", "content"}]}``), which keeps multi-turn
    probes intact. Older reports wrote the prompt as a bare string or a single message.
    """
    if isinstance(prompt, dict) and isinstance(prompt.get("turns"), list):
        steps: list[tuple[str, str]] = []
        for turn in prompt["turns"]:
            if not isinstance(turn, dict):
                continue
            source = _ATIF_SOURCE_BY_ROLE.get(str(turn.get("role")), "user")
            steps.append((source, _message_text(turn.get("content"))))
        return steps
    return [("user", _message_text(prompt))]


def garak_test_case_name(probe: str, detector: str) -> str:
    """Test case a probe and detector pair publishes under."""
    return f"garak/{probe}/{detector.removeprefix('detector.')}"


# --- Building sessions --------------------------------------------------------


def _base_session(
    *,
    session_id: str,
    probe: str,
    detector: str,
    rewards: dict[str, float],
    garak_details: dict[str, Any],
    report: GarakReport,
    spec: IntakePublicationSpec,
    agent_name: str,
    model_name: str | None,
) -> AtifCreateParams:
    agent: AtifAgentParam = {"name": agent_name, "version": spec.agent_version}
    if model_name is not None:
        agent["model_name"] = model_name
    details = {"probe": probe, "detector": detector.removeprefix("detector."), **garak_details}
    if report.garak_version is not None:
        details["version"] = report.garak_version
    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id,
        "agent": agent,
        "evaluation_context": {
            "evaluation_name": spec.evaluation_id,
            "test_case_name": garak_test_case_name(probe, detector),
        },
        "extra": {
            "task_id": f"garak/{probe}",
            "verifier_result": {"rewards": rewards},
            "garak": details,
        },
    }


def attempt_sessions(
    report: GarakReport,
    *,
    spec: IntakePublicationSpec,
    agent_name: str,
    model_name: str | None,
    eval_threshold: float,
) -> tuple[list[AtifCreateParams], int]:
    """One ATIF session per scored output, and how many outputs garak left unscored.

    An attempt holds one output per generation and one score per output per detector, so an attempt
    with two detectors and three generations yields six sessions. A score of ``None`` means the
    detector did not judge that output (an empty response, usually); garak leaves those out of its
    pass rate and so does this.

    A score passes when it is below ``eval_threshold``, which is how garak's own evaluator reads it.
    """
    timestamp = report.started_at.isoformat()
    sessions: list[AtifCreateParams] = []
    unscored = 0
    for attempt in report.attempts:
        probe = str(attempt.get("probe_classname") or "unknown")
        attempt_id = str(attempt.get("uuid") or f"{probe}:{attempt.get('seq')}")
        outputs = attempt.get("outputs")
        outputs = outputs if isinstance(outputs, list) else []
        prompt_steps = _prompt_steps(attempt.get("prompt"))
        detector_results = attempt.get("detector_results")
        if not isinstance(detector_results, dict):
            continue
        for detector, scores in detector_results.items():
            if not isinstance(scores, list):
                continue
            for output_index, score in enumerate(scores):
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    unscored += 1
                    continue
                body = _base_session(
                    # The evaluation is part of the id so the same audit can be published under two
                    # Evaluations without one replacing the other's rows.
                    session_id=f"garak:{spec.evaluation_id}:{attempt_id}:{detector}:{output_index}",
                    probe=probe,
                    detector=detector,
                    rewards={
                        PASS_RATE_REWARD: 1.0 if float(score) < eval_threshold else 0.0,
                        DETECTOR_SCORE_REWARD: float(score),
                    },
                    garak_details={
                        "attempt_uuid": attempt_id,
                        "attempt_seq": attempt.get("seq"),
                        "output_index": output_index,
                        "goal": attempt.get("goal"),
                        "eval_threshold": eval_threshold,
                    },
                    report=report,
                    spec=spec,
                    agent_name=agent_name,
                    model_name=model_name,
                )
                output_text = _message_text(outputs[output_index]) if output_index < len(outputs) else ""
                steps: list[AtifStepParam] = [
                    # Every step carries the run's start time: Intake keys a session's rows on it, so
                    # a step left to the server's ingest clock would duplicate on re-publish.
                    {"source": source, "step_id": step_id, "message": text, "timestamp": timestamp}
                    for step_id, (source, text) in enumerate(prompt_steps, start=1)
                ]
                steps.append(
                    {"source": "agent", "step_id": len(steps) + 1, "message": output_text, "timestamp": timestamp}
                )
                body["steps"] = steps
                sessions.append(body)
    return sessions, unscored


def probe_sessions(
    report: GarakReport,
    *,
    spec: IntakePublicationSpec,
    agent_name: str,
    model_name: str | None,
) -> tuple[list[AtifCreateParams], int]:
    """One ATIF session per probe and detector, and how many pairs had nothing evaluated."""
    timestamp = report.started_at.isoformat()
    sessions: list[AtifCreateParams] = []
    unscored = 0
    for entry in report.evals:
        probe = str(entry.get("probe") or "unknown")
        detector = str(entry.get("detector") or "unknown")
        passed = entry.get("passed")
        # `total_evaluated` leaves out the outputs the detector did not score; older reports only
        # have `total`.
        total = entry.get("total_evaluated", entry.get("total"))
        if not isinstance(passed, int) or not isinstance(total, int) or total <= 0:
            unscored += 1
            continue
        body = _base_session(
            session_id=f"garak:{spec.evaluation_id}:{report.run_id}:{probe}:{detector}",
            probe=probe,
            detector=detector,
            rewards={PASS_RATE_REWARD: passed / total},
            garak_details={"passed": passed, "total_evaluated": total, "nones": entry.get("nones")},
            report=report,
            spec=spec,
            agent_name=agent_name,
            model_name=model_name,
        )
        body["steps"] = [
            {
                "source": "agent",
                "step_id": 1,
                "message": f"{passed} of {total} outputs passed {detector.removeprefix('detector.')}",
                "timestamp": timestamp,
            }
        ]
        sessions.append(body)
    return sessions, unscored


# --- Publishing ---------------------------------------------------------------


async def publish_report_to_intake(
    report: GarakReport,
    *,
    intake: AsyncIntakeClient,
    spec: IntakePublicationSpec,
    workspace: str,
    agent_name: str,
    model_name: str | None,
    eval_threshold: float,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> PublicationOutcome:
    """Publish ``report`` under the Evaluation ``spec`` names and describe what landed.

    Publishing is not atomic and Intake has no rollback, so one session failing does not stop the
    others: every session that can land does, and the failures are reported together. Re-publishing
    is idempotent, because each session id and timestamp is a function of the report and not of when
    it was published.

    Raises ``NotFoundError`` when the Evaluation does not exist.
    """
    # ATIF ingest refuses an unknown Evaluation once per session. Checking first turns a typo into
    # one clear stop instead of that many failed writes.
    await intake.get_evaluation(name=spec.evaluation_id, workspace=workspace)

    if spec.granularity == "attempt":
        sessions, unscored = attempt_sessions(
            report, spec=spec, agent_name=agent_name, model_name=model_name, eval_threshold=eval_threshold
        )
    else:
        sessions, unscored = probe_sessions(report, spec=spec, agent_name=agent_name, model_name=model_name)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _publish(body: AtifCreateParams) -> None:
        async with semaphore:
            await intake.create_atif(workspace=workspace, body=AtifCreateRequest(root=body))

    results = await asyncio.gather(*(_publish(body) for body in sessions), return_exceptions=True)
    failures = [
        (body["session_id"], result)
        for body, result in zip(sessions, results, strict=True)
        if isinstance(result, BaseException)
    ]
    published = len(sessions) - len(failures)
    if failures:
        shown = "; ".join(
            f"{session_id}: {type(error).__name__}: {error}" for session_id, error in failures[:_MAX_REPORTED_FAILURES]
        )
        more = len(failures) - _MAX_REPORTED_FAILURES
        return PublicationOutcome(
            status=HelixJobStatus.ERROR,
            evaluation_id=spec.evaluation_id,
            granularity=spec.granularity,
            session_count=published,
            unscored_count=unscored,
            error=f"{len(failures)} of {len(sessions)} session(s) failed to publish ({published} succeeded). "
            f"{shown}" + (f"; and {more} more" if more > 0 else ""),
        )
    if not sessions:
        return PublicationOutcome(
            status=HelixJobStatus.ERROR,
            evaluation_id=spec.evaluation_id,
            granularity=spec.granularity,
            unscored_count=unscored,
            error="The garak report has no scored results to publish.",
        )
    return PublicationOutcome(
        status=HelixJobStatus.COMPLETED,
        evaluation_id=spec.evaluation_id,
        granularity=spec.granularity,
        session_count=published,
        unscored_count=unscored,
    )


def _run_in_fresh_loop(make_coroutine: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine to completion on a thread of its own.

    The audit job is synchronous, and its caller may or may not already be inside an event loop (a
    notebook is). A dedicated thread makes ``asyncio.run`` safe in both cases.
    """

    def _run() -> T:
        return asyncio.run(make_coroutine())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


async def _with_intake(
    client: NemoClient | None,
    async_client: AsyncNemoClient | None,
    use: Callable[[AsyncIntakeClient], Awaitable[T]],
) -> T:
    """Call ``use`` with an Intake client carrying the platform client's identity, on this event loop.

    Neither injected client's transport can be used directly: the sync one is not async, and the
    async one is bound to the caller's loop. So the sync client is converted, and the async client is
    cloned onto a throwaway transport that keeps its request hooks (auth refresh lives there).
    """
    if client is not None:
        async with client.to_async() as converted:
            return await use(AsyncIntakeClient.from_client(converted))
    assert async_client is not None
    event_hooks = {event: list(hooks) for event, hooks in async_client._client.event_hooks.items()}
    async with AsyncClient(timeout=async_client._client.timeout, event_hooks=event_hooks) as http_client:
        return await use(AsyncIntakeClient.from_client(async_client.with_http_client(http_client)))


def check_evaluation_exists(
    spec: IntakePublicationSpec,
    *,
    workspace: str,
    client: NemoClient | None,
    async_client: AsyncNemoClient | None,
) -> str | None:
    """Return why the Evaluation cannot be published to, or ``None`` when it is there.

    Called before the probes run, so a mistyped Evaluation name costs seconds and not a whole audit.
    """
    if client is None and async_client is None:
        return "No platform client available; cannot publish to Intake."

    async def _check(intake: AsyncIntakeClient) -> None:
        await intake.get_evaluation(name=spec.evaluation_id, workspace=workspace)

    try:
        _run_in_fresh_loop(lambda: _with_intake(client, async_client, _check))
    except NotFoundError:
        return _missing_evaluation_message(spec.evaluation_id, workspace)
    except Exception as error:
        return f"Could not read evaluation {spec.evaluation_id!r}: {type(error).__name__}: {error}"
    return None


def _missing_evaluation_message(evaluation_id: str, workspace: str) -> str:
    return (
        f"Evaluation {evaluation_id!r} does not exist in workspace {workspace!r}. "
        "Create it before running the audit; the audit does not create it."
    )


def publish_audit_report(
    report_path: Path,
    *,
    spec: IntakePublicationSpec,
    workspace: str,
    target_name: str,
    model_name: str | None,
    eval_threshold: float,
    client: NemoClient | None = None,
    async_client: AsyncNemoClient | None = None,
) -> PublicationOutcome:
    """Publish the garak report at ``report_path`` to Intake and describe what happened.

    Raises :class:`PublicationFailedError` when publication fails and ``spec.required`` is set;
    otherwise returns a failed outcome for the audit output. The report itself is already saved, so a
    failure costs a re-publish (call this again with the saved ``report.jsonl``) and not a re-run.

    Pass whichever platform client is at hand. ``target_name`` is the audit target's name, used as the
    agent name unless the spec names one; ``model_name`` is the model the target points at.
    """

    def fail(error: str, session_count: int = 0, unscored_count: int = 0) -> PublicationOutcome:
        outcome = PublicationOutcome(
            status=HelixJobStatus.ERROR,
            evaluation_id=spec.evaluation_id,
            granularity=spec.granularity,
            session_count=session_count,
            unscored_count=unscored_count,
            error=error,
        )
        if spec.required:
            raise PublicationFailedError(outcome)
        logger.warning(
            "Publication to Intake failed for evaluation %r but was not required; continuing: %s",
            spec.evaluation_id,
            error,
        )
        return outcome

    if client is None and async_client is None:
        return fail("No platform client available; cannot publish to Intake.")

    try:
        report = read_report(report_path)
    except (OSError, ValueError) as error:
        return fail(str(error))

    logger.info(
        "Publishing garak report to Intake under evaluation %r in workspace %r (%s granularity)",
        spec.evaluation_id,
        workspace,
        spec.granularity,
    )
    try:
        outcome = _run_in_fresh_loop(
            lambda: _with_intake(
                client,
                async_client,
                lambda intake: publish_report_to_intake(
                    report,
                    intake=intake,
                    spec=spec,
                    workspace=workspace,
                    agent_name=spec.agent_name or target_name,
                    model_name=model_name,
                    eval_threshold=eval_threshold,
                ),
            )
        )
    except NotFoundError:
        return fail(_missing_evaluation_message(spec.evaluation_id, workspace))
    except NemoClientError as error:
        return fail(f"{type(error).__name__}: {error}")
    except Exception as error:
        # `required=False` promises the audit survives a failed publish. Letting an unforeseen error
        # escape would break that promise for exactly the failures nobody anticipated.
        logger.exception("Unexpected error publishing to Intake for evaluation %r", spec.evaluation_id)
        return fail(f"Unexpected {type(error).__name__}: {error}")

    if outcome.status != HelixJobStatus.COMPLETED:
        return fail(outcome.error or "publication failed", outcome.session_count, outcome.unscored_count)
    logger.info(
        "Published %d session(s) to Intake under evaluation %r (%d unscored result(s) left out)",
        outcome.session_count,
        spec.evaluation_id,
        outcome.unscored_count,
    )
    return outcome
