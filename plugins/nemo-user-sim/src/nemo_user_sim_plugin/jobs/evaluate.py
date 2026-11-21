# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scores a simulate job's conversations with UserSim's judges and marks which pass.

Runs as the job's second step, reading the trajectories the first step left in the job's storage.
"""

import json
from typing import Any, ClassVar

import pandas as pd
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.explain import picked, what_was_tested, why_it_failed
from nemo_user_sim_plugin.jobs._common import (
    EVALUATIONS,
    PROBE_SCORERS,
    PROFILES,
    SCORES,
    SCORING_ERROR,
    TRAJECTORIES,
    read_result_frame,
    usersim_models,
)
from nemo_user_sim_plugin.jobs.spec import EvaluateSpec
from pydantic import BaseModel
from usersim.cli._pipeline import evaluate_dataframe
from usersim.engine.models_file import load_models_config
from usersim.selection import select_trajectories
from usersim.selection.engine import COL_AXIS_SCORES, COL_FAILED_GATES, COL_QUALITY

# Selection gates that judge the simulation rather than the agent: no judge output, a skipped or failed
# simulation, too few turns. Health gates (``health:*``) and judge disagreement carry a prefix.
_SIMULATION_GATES = {"no_eval", "skipped", "status", "min_turns"}
_SCORER_ERROR = ":scorer_error"


def _cell(value: Any) -> Any:
    """Selection columns arrive as JSON text from parquet and as objects in memory."""
    return json.loads(value) if isinstance(value, str) else value


def fully_scored(selection_row: dict) -> bool:
    """Whether the judge scored the conversation and no scorer it needed failed.

    UserSim records a scorer that failed as a ``<axis>:scorer_error`` gate and leaves its scores out, so a
    conversation can clear every other gate without having been scored on that axis.
    """
    failed_gates, axis_scores = (_cell(selection_row[column]) for column in (COL_FAILED_GATES, COL_AXIS_SCORES))
    return bool(axis_scores) and not any(gate.endswith(_SCORER_ERROR) for gate in failed_gates)


def agent_verdict(selection_row: dict) -> dict:
    """Whether the agent passed, from the judge's quality axes only.

    UserSim's selection also rejects conversations for simulator problems, such as the simulated user
    breaking character. Those decide training data, not the agent's result. ``passed`` is None when the
    conversation wasn't fully scored and no axis it was scored on failed.
    """
    failed_gates = _cell(selection_row[COL_FAILED_GATES])
    failed_axes = [gate for gate in failed_gates if ":" not in gate and gate not in _SIMULATION_GATES]
    if not _cell(selection_row[COL_AXIS_SCORES]):
        passed = None
    elif failed_axes:
        passed = False
    else:
        passed = True if fully_scored(selection_row) else None
    return {"passed": passed, "reason": ", ".join(failed_axes) or None}


def scoring_failure(error: Exception) -> str:
    """Why the judge couldn't score, as one sentence. Data Designer boxes its errors; keep only the cause."""
    text = str(error)
    cause = next(
        (line.split("Cause:", 1)[1].strip() for line in text.splitlines() if "Cause:" in line),
        text.strip().splitlines()[0] if text.strip() else type(error).__name__,
    )
    return f"Scoring failed, so no conversation was judged. {cause}"


class EvaluateJob(NemoJob):
    name: ClassVar[str] = "evaluate"
    description: ClassVar[str] = "Score simulated conversations with UserSim's judges."
    spec_schema: ClassVar[type[BaseModel]] = EvaluateSpec
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict:
        spec = EvaluateSpec.model_validate(config)
        work = ctx.storage.persistent
        trajectories = work / TRAJECTORIES
        if not trajectories.exists():  # every conversation failed, so there's nothing to score
            return {"status": "completed", "scored": 0, "passed": 0}
        labels: dict[str, str] = {}
        if spec.scenarios:  # the safety_agentic scorer reads the run's bank and actions
            fetched = scenarios.fetch(FilesClient.from_client(client), ctx.workspace, spec.scenarios)
            scenarios.install(fetched, work / "scenarios")
            labels = scenarios.outcome_labels(fetched)
        conversations = read_result_frame(trajectories)
        models = load_models_config(usersim_models(client, ctx.workspace, work, assistant_model=spec.assistant_model))
        try:
            scorers = sorted({PROBE_SCORERS[probe] for probe in conversations.probe_type if probe in PROBE_SCORERS})
            evaluations = evaluate_dataframe(conversations, models=models, scorers=scorers)
        except Exception as error:
            # The jobs service reports only the exit code, so leave the reason where Studio and the CLI look.
            (work / SCORING_ERROR).write_text(json.dumps({"message": scoring_failure(error)}))
            ctx.results.save("scoring_error", work / SCORING_ERROR)
            raise
        evaluations.to_parquet(work / EVALUATIONS, index=False)
        ctx.results.save("evaluations", work / EVALUATIONS)

        selection = select_trajectories(conversations, evaluations, profile=PROFILES["strict"]).all
        cells = {
            row["trajectory_id"]: (_cell(row["assistant_eval"]) or {}, row.get("locale") or "en_US")
            for row in evaluations.to_dict("records")
        }
        scores = {}
        for row in selection.to_dict("records"):
            verdict = agent_verdict(row)
            cell, locale = cells.get(row["trajectory_id"], ({}, "en_US"))
            scores[row["trajectory_id"]] = {
                **verdict,
                "quality": None if pd.isna(row[COL_QUALITY]) else float(row[COL_QUALITY]),
                "tested": what_was_tested(cell, locale),
                "picked": picked(cell, labels),
                "why": why_it_failed(cell, verdict["reason"].split(", ")) if verdict["reason"] else {},
            }
        (work / SCORES).write_text(json.dumps(scores))
        ctx.results.save("scores", work / SCORES)
        return {
            "status": "completed",
            "scored": len(scores),
            "passed": sum(s["passed"] is True for s in scores.values()),
        }
