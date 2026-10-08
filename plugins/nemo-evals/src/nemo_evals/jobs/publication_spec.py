# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a completed evaluation publishes its results, beyond its own result bundle.

Shared by both evaluation jobs — the agent-eval spec and the dataset-driven (row) eval spec — so it
lives here rather than in either one. The row variants add the one field that only makes sense for a
dataset: which column identifies a test case.
"""

from __future__ import annotations

from nemo_evals.intake.mapping import DEFAULT_AGENT_VERSION
from pydantic import BaseModel, ConfigDict, Field


class IntakePublicationSpec(BaseModel):
    """Publish this run's trials and scores to Intake, under an Evaluation that already exists.

    ``evaluation_id`` is the *name* of a ``client.evaluations`` record. Intake stores that record as
    its ``Experiment`` entity and the SDK's ``publish_to_intake`` calls the argument
    ``experiment_id``, but the value is the same one either way — the parent ``client.experiments``
    group is a different resource and is not what goes here. The job never creates the Evaluation: a
    missing one is an error, because nothing in an eval spec can supply the dataset identity
    ``evaluations.create`` requires.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(
        min_length=1,
        description="Name of the existing Evaluation to publish under. Must already exist; the job does not create it.",
    )
    # Both are `min_length=1` because an empty string is worse than omitting them: `agent_name=""`
    # is not None, so it satisfies the identity validator and skips the derivation it exists to
    # require, then resolves back to "" at publish time — the empty agent name the validator was
    # written to prevent. Omit the field to get derivation; there is no use for a blank one.
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        description="Agent name recorded on each published trajectory. Derived from the target when "
        "it names one; required otherwise.",
    )
    agent_version: str = Field(
        default=DEFAULT_AGENT_VERSION,
        min_length=1,
        description="Agent version recorded on each published trajectory. Neither a Model nor an "
        "Agent carries a version, so this defaults to 'unknown' unless the submitter supplies one.",
    )
    required: bool = Field(
        default=True,
        description="Fail the job when publication fails. Defaults to True so a run that asked to "
        "publish does not report success with nothing in Experiments. The result bundle is saved "
        "before publication runs, so a failed job still leaves the results intact to re-publish. "
        "Set False to keep the job successful and report the failure in its output instead.",
    )


class PublicationSpec(BaseModel):
    """Where a completed agent-evaluation run publishes its results."""

    model_config = ConfigDict(extra="forbid")

    intake: IntakePublicationSpec | None = Field(
        default=None, description="Publish trials and scores to Intake. Omit to publish nowhere."
    )


class RowIntakePublicationSpec(IntakePublicationSpec):
    """Intake publication for a dataset-driven evaluation, where a trial is a dataset row."""

    test_case_id_field: str | None = Field(
        default=None,
        min_length=1,
        description="Dataset column identifying each row, recorded as the published test case id. "
        "Defaults to a hash of the row's content, which keeps a row's id stable across dataset "
        "reorderings and revisions so the same test case can be compared run over run; editing a "
        "row makes it a new test case. Name a column here when the dataset has a real identifier — "
        "it reads better and survives content edits. Values must be unique per row; the run is "
        "rejected if they are not.",
    )


class RowPublicationSpec(BaseModel):
    """Where a completed dataset-driven evaluation publishes its results."""

    model_config = ConfigDict(extra="forbid")

    intake: RowIntakePublicationSpec | None = Field(
        default=None, description="Publish scored rows to Intake. Omit to publish nowhere."
    )
