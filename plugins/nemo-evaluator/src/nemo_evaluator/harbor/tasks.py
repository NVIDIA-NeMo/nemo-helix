# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Harbor job inputs and internal, verified member definitions."""

import re
from typing import Annotated, Literal, Self

from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef, TasksetRef, parse_subentity_ref
from nemo_evaluator.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UnsupportedTaskKindError(ValueError):
    """A stored task cannot run on the requested target."""


def require_task_kind(entity_name: str, stored_kind: str, target_kind: str) -> None:
    expected = "harbor" if target_kind == "harbor" else "evaluator"
    if stored_kind != expected:
        raise UnsupportedTaskKindError(
            f"Task '{entity_name}' has stored kind {stored_kind!r}; target {target_kind!r} accepts {expected!r}."
        )


def require_pin(ref: TaskRef | TasksetRef) -> None:
    """Canonical references are qualified and immutable, unlike public selectors."""

    name, separator, digest = ref.root.partition("#")
    if "/" not in name or not separator or re.fullmatch(DIGEST_PATTERN, digest) is None:
        raise ValueError(f"Expected a qualified immutable digest reference, got {ref.root!r}")


def qualified_task_refs(refs: list[TaskRef], workspace: str) -> list[TaskRef]:
    """Qualify selectors and reject repeated entities before any store access."""
    seen: set[tuple[str, str]] = set()
    qualified = []
    for ref in refs:
        member_workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
        identity = (member_workspace, name)
        if identity in seen:
            raise ValueError(f"Duplicate task identity: {member_workspace}/{name}")
        seen.add(identity)
        qualified.append(TaskRef(f"{member_workspace}/{name}#{fragment}"))
    if not qualified:
        raise ValueError("Expected at least one stored task")
    return qualified


class PinnedHarborTaskset(BaseModel):
    """Canonical job source pointing to an existing Taskset revision.

    This is a deferred job input, not another stored taskset schema. The worker
    resolves its pinned members and materializes their Harbor archives locally.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["harbor-taskset"] = "harbor-taskset"
    taskset_ref: TasksetRef = Field(
        description="Existing taskset revision in workspace/name#<sha256-digest> form. "
        "The reference must be qualified and digest-pinned; mutable tags are not accepted."
    )

    @field_validator("taskset_ref")
    @classmethod
    def _pin(cls, value: TasksetRef) -> TasksetRef:
        require_pin(value)
        return value


class PinnedHarborTaskList(BaseModel):
    """Canonical job source selecting exact task revisions without creating a taskset.

    The worker resolves these references and materializes their Harbor archives
    locally, preserving the requested task order.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["harbor-task-list"] = "harbor-task-list"
    task_refs: list[TaskRef] = Field(
        min_length=1,
        description="Task revisions in execution order, each in workspace/name#<sha256-digest> form. "
        "References must be qualified and digest-pinned; mutable tags and repeated task identities are not accepted.",
    )

    @field_validator("task_refs")
    @classmethod
    def _pins(cls, value: list[TaskRef]) -> list[TaskRef]:
        for ref in value:
            require_pin(ref)
        return qualified_task_refs(value, "default")


PinnedHarborSource = Annotated[PinnedHarborTaskset | PinnedHarborTaskList, Field(discriminator="kind")]


class StoredHarborTask(BaseModel):
    """Runtime descriptor; never embedded in canonical jobs."""

    model_config = ConfigDict(extra="forbid")
    entity_name: str
    revision_digest: str = Field(pattern=DIGEST_PATTERN, min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH)
    definition: HarborTaskDefinition

    @model_validator(mode="after")
    def _validate_identity_and_archive(self) -> Self:
        if "#" in self.entity_name or "/" not in self.entity_name:
            raise ValueError("entity_name must be qualified without a revision fragment")
        TaskRef(self.entity_name)
        self.definition = HarborTaskDefinition.model_validate(self.definition.model_dump())
        return self
