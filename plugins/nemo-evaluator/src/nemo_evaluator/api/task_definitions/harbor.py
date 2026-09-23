# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Harbor task kind: a packaged task directory, run and scored by Harbor."""

from __future__ import annotations

from typing import Any, Literal

from nemo_evaluator.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_helix_plugin.refs import FILESET_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field


class HarborTaskDefinition(BaseModel):
    """A reference to the task's packaged files, plus a projection of Harbor's own config.

    Harbor identifies a task by a *directory* — ``task.toml``, an instruction, an environment — so
    what is stored is a reference to that directory's archive in the Files service, not the files
    themselves. One fileset per task, so a task shared by several tasksets is stored once. The
    archive is materialized back into ``<dir>/<task-name>/`` at run time, which is the layout
    Harbor's own discovery expects.

    Which agent runs the task is *not* stored here. That comes from the run's target
    (``HarborRunnerTarget``), so the same stored task can be evaluated against different agents.
    Harbor's own ``[agent]`` block — carried inside ``config`` — configures how the agent *phase*
    runs (timeout, user, network policy), not which agent it is.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["harbor"] = Field(description="Task kind discriminator.")
    archive_ref: str = Field(
        pattern=FILESET_REF_PATTERN,
        description="Files reference to the task's packaged directory (format: workspace/fileset#path).",
    )
    archive_digest: str = Field(
        description="Content hash Harbor computed over the task directory. This is the authoritative "
        "identity of a Harbor task's content — every file, including task.toml.",
        min_length=DIGEST_LENGTH,
        max_length=DIGEST_LENGTH,
        pattern=DIGEST_PATTERN,
    )
    instruction: str | None = Field(
        default=None, description="The task's instruction text, when it has one (multi-step tasks may not)."
    )
    # Excluded from the revision digest (see ``_DERIVED_SPEC_FIELDS`` in ``entities``). Safe only
    # because this is never an execution input: Harbor reads the real ``task.toml`` out of the
    # materialized archive, and ``archive_digest`` already covers every file in that directory.
    # Hashing the projection too would add no coverage, and would make revision history sensitive to
    # Harbor's serialization — a release that reordered keys would cut a revision for byte-identical
    # files. Anything here that becomes a genuine execution or grading input must be digested.
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Harbor's own task configuration (verifier, agent, environment, steps), as published. "
        "A queryable projection of task.toml — inspect a task's verifier without downloading the "
        "archive. Opaque here: Harbor owns this schema.",
    )
