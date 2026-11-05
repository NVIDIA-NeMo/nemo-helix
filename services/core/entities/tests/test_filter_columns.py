# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapped-field validation and timestamp operands against real SQL queries."""

from datetime import date, datetime, timedelta, timezone

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.entities.app.repository.exceptions import InvalidFilterError
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity, DBWorkspace
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, aliased


@pytest.mark.parametrize(
    "model,field",
    [
        (DBEntity, "__class__"),
        (DBEntity, "data."),
        (DBEntity, "data..key"),
        (aliased(DBEntity), "metadata"),
        (aliased(DBEntity), "missing"),
        (DBWorkspace, "data.key"),
    ],
)
def test_noncolumn_filter_rejected(model, field):
    """Cover distinct invalid lookup branches; HTTP tests cover error translation."""
    with pytest.raises(InvalidFilterError, match="Invalid filter field"):
        ComparisonOperation(field=field, operator=FilterOperator.EQ, value="1").apply(SQLAlchemyFilterRepository(model))


@pytest.fixture
def stored_entities():
    """Two distinguishable rows expose accidental base-table/alias cross joins."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            session.add(DBWorkspace(id="workspace-default", name="default"))
            session.add_all(
                [
                    DBEntity(
                        id="model-target",
                        workspace="default",
                        entity_type="model",
                        name="target",
                        data={"nested": {"key": "chosen"}},
                        created_at=datetime(2026, 1, 1),
                    ),
                    DBEntity(
                        id="model-other",
                        workspace="default",
                        entity_type="model",
                        name="other",
                        data={"nested": {"key": "other"}},
                        created_at=datetime(2026, 1, 2),
                    ),
                ]
            )
            session.commit()
            yield session
    finally:
        engine.dispose()


@pytest.mark.parametrize("field,value", [("name", "target"), ("data.nested.key", "chosen")])
def test_filter_uses_queried_alias(stored_entities: Session, field, value):
    """A filter accidentally bound to DBEntity would return both aliased names."""
    model = aliased(DBEntity)
    expression = ComparisonOperation(field=field, operator=FilterOperator.EQ, value=value).apply(
        SQLAlchemyFilterRepository(model)
    )
    assert stored_entities.scalars(select(model.name).where(expression)).all() == ["target"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (datetime(2026, 1, 1), ["target"]),
        (datetime(2026, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))), ["target"]),
        (datetime(2026, 1, 1, 2), []),
        # Preserve SQLite's existing date-only binding semantics (no epoch conversion).
        (date(2026, 1, 1), []),
        (None, []),
    ],
)
def test_python_timestamp_operands_match_stored_rows(stored_entities: Session, value, expected):
    """Preserve supported Python values, normalizing only aware datetimes."""
    expression = ComparisonOperation(field="created_at", operator=FilterOperator.EQ, value=value).apply(
        SQLAlchemyFilterRepository(DBEntity)
    )
    assert stored_entities.scalars(select(DBEntity.name).where(expression)).all() == expected


@pytest.mark.parametrize("value", [42, True, {}, []])
def test_unsupported_timestamp_operand_rejected(value):
    """Do not accept numeric epochs or other JSON values as timestamps."""
    with pytest.raises(InvalidFilterError, match="Invalid timestamp filter value"):
        ComparisonOperation(field="created_at", operator=FilterOperator.EQ, value=value).apply(
            SQLAlchemyFilterRepository(DBEntity)
        )
