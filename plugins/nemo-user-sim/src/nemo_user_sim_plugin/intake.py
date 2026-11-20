# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Intake publication and reads of reviewer feedback.

Publication is structural, as in the Helix Evals plugin: callers pass no client when the spec didn't
ask for it. A failure is reported per conversation and never fails the job.
"""

import logging
from typing import Any

from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.intake.types import AtifCreateRequest, EvaluationCreateRequest, ExperimentCreateRequest

logger = logging.getLogger(__name__)


async def register_run(intake: Any, workspace: str, run: str, *, locale: str, target: str) -> bool:
    """Tag a run in Intake: an Evaluation named after it, in the target's Experiment. True when tagging works.

    One Experiment per tested agent or model lets Studio's Experiments view compare its runs over time.
    A retried attempt finds the Evaluation already there. Any other failure leaves the conversations
    untagged rather than failing the run, since publication is optional.
    """
    try:
        experiment = await intake.create_experiment(
            workspace=workspace,
            body=ExperimentCreateRequest(name=f"user-sim-{target}", description=f"UserSim runs against {target}"),
            exist_ok=True,
        )
        body = EvaluationCreateRequest(
            name=run,
            experiment_ids=[experiment.data().id],
            dataset_name=f"nemotron-personas-{locale}",
            description=f"UserSim run {run} against {target}",
            metadata={"source": "user-sim", "target": target},
        )
        try:
            await intake.create_evaluation(workspace=workspace, body=body)
        except ConflictError:
            pass
    except Exception as exc:
        logger.warning("Couldn't register run %s with Intake; its conversations won't be tagged: %s", run, exc)
        return False
    return True


async def publish_conversation(intake: Any | None, workspace: str, payload: dict) -> dict:
    session_id = payload["session_id"]
    if intake is None:
        return {"status": "skipped", "session_id": session_id, "reason": "not requested"}
    try:
        await intake.create_atif(workspace=workspace, body=AtifCreateRequest(payload))
    except Exception as exc:
        logger.warning("Intake publication failed for %s: %s", session_id, exc)
        return {"status": "failed", "session_id": session_id, "reason": str(exc)}
    return {"status": "published", "session_id": session_id, "reason": None}


async def review_feedback(intake: Any, workspace: str, session_ids: set[str]) -> dict[str, str]:
    """A reviewer's latest Studio feedback, ``positive`` or ``negative``, per session in ``session_ids``.

    Studio records a thumbs click as a ``feedback`` annotation. An Intake error propagates: exporting
    without the reviews would ship conversations a reviewer rejected.
    """
    query = {"filter": {"kind": "feedback"}, "sort": "created_at"}
    feedback: dict[str, str] = {}
    async for annotation in await intake.list_annotations(workspace=workspace, query_params=query):
        if annotation.session_id in session_ids:
            feedback[annotation.session_id] = str(annotation.value)
    return feedback
