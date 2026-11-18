# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
from nemo_helix_plugin.filter_ops import (
    ComparisonOperation,
    FilterOperator,
    FilterRepository,
    LogicalOperation,
    RelationshipOperation,
)


def test_relationship_operation_serializes_to_the_entities_service_shape() -> None:
    operation = LogicalOperation(
        operator=FilterOperator.OR,
        operations=[
            RelationshipOperation(
                relationship="revisions",
                condition=ComparisonOperation(field="frontmatter.tags", operator=FilterOperator.CONTAINS, value="pdf"),
            ),
            RelationshipOperation(relationship="revisions"),
        ],
    )

    assert operation.to_dict() == {
        "$or": [{"revisions": {"frontmatter.tags": {"$contains": "pdf"}}}, {"revisions": {"$exists": True}}]
    }


def test_relationship_operation_cannot_be_evaluated_in_memory() -> None:
    with pytest.raises(NotImplementedError, match="entities service"):
        RelationshipOperation(relationship="revisions").apply(cast(FilterRepository, object()))
