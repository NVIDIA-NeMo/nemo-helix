# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only mapped filter columns and valid JSON paths are accepted, including aliases."""

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.core.entities.app.repository.exceptions import InvalidFilterError
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity, DBWorkspace
from sqlalchemy import select
from sqlalchemy.orm import aliased


@pytest.mark.parametrize("model", [DBEntity, DBWorkspace, aliased(DBEntity), aliased(DBWorkspace)])
@pytest.mark.parametrize("field", ["metadata", "registry", "__class__", "to_pydantic", "missing", "data.", "data..key"])
def test_noncolumn_filter_rejected(model, field):
    repository = SQLAlchemyFilterRepository(model)
    with pytest.raises(InvalidFilterError, match="Invalid filter field"):
        ComparisonOperation(field=field, operator=FilterOperator.EQ, value="1").apply(repository)


@pytest.mark.parametrize("model", [DBEntity, DBWorkspace, aliased(DBEntity), aliased(DBWorkspace)])
def test_mapped_filter_uses_model_alias(model):
    repository = SQLAlchemyFilterRepository(model)
    column, is_json = repository._get_column("name")
    assert column is model.name
    assert not is_json
    expression = ComparisonOperation(field="name", operator=FilterOperator.EQ, value="test").apply(repository)
    statement = select(model).where(expression)
    assert len(statement.get_final_froms()) == 1


@pytest.mark.parametrize("model", [DBEntity, aliased(DBEntity)])
def test_json_data_path_supported(model):
    repository = SQLAlchemyFilterRepository(model)
    column, is_json = repository._get_column("data.nested.key")
    assert is_json
    assert len(select(model).where(column == "value").get_final_froms()) == 1


def test_workspace_has_no_json_data_column():
    with pytest.raises(InvalidFilterError, match="Invalid filter field"):
        SQLAlchemyFilterRepository(DBWorkspace)._get_column("data.key")
