# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored Harbor descriptors consumed by archive materialization."""

from typing import Self

from nemo_evaluator.api.schemas import HarborTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from pydantic import model_validator


class StoredHarborTask(TaskProvenance):
    """Verified published member used by archive materialization on the job worker and submission for validation."""

    definition: HarborTaskDefinition

    @model_validator(mode="after")
    def _validate_definition(self) -> Self:
        self.definition = HarborTaskDefinition.model_validate(self.definition.model_dump())
        return self
