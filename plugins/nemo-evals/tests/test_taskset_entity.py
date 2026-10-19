# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialization round-trip tests for TasksetEntity.

The entity store persists custom fields with ``model_dump(exclude=base, mode="json")`` into a JSON
column and rebuilds them with ``model_validate``. This exercises that round-trip for the fields that
carry non-trivial nested types — the task references and metadata.
"""

from __future__ import annotations

import json

from nemo_evals.api.schemas import TaskRef
from nemo_evals.entities import TasksetEntity


def _entity() -> TasksetEntity:
    return TasksetEntity(
        name="ts-1",
        workspace="default",
        description="A smoke-test grouping.",
        # A workspace-qualified ref and a bare name.
        tasks=[TaskRef("default/task-a"), TaskRef("task-b")],
        metadata=[{"key": "suite", "value": "smoke"}],
    )


def _roundtrip(entity: TasksetEntity) -> TasksetEntity:
    data = entity.model_dump(exclude=TasksetEntity.__base_fields__, exclude_computed_fields=True, mode="json")
    data = json.loads(json.dumps(data))  # prove JSON-serializable (the store uses a JSON column)
    return TasksetEntity.model_validate({"name": entity.name, "workspace": entity.workspace, **data})


def test_roundtrip_preserves_taskset_fields() -> None:
    entity = _entity()

    restored = _roundtrip(entity)

    assert restored.description == "A smoke-test grouping."
    assert [(m.key, m.value) for m in restored.metadata] == [("suite", "smoke")]
    # Task refs survive as RootModel strings.
    assert isinstance(restored.tasks[0], TaskRef)
    assert restored.tasks[0].root == "default/task-a"
    assert restored.tasks[1].root == "task-b"


def test_entity_type_is_taskset() -> None:
    assert TasksetEntity.__entity_type__ == "taskset"
