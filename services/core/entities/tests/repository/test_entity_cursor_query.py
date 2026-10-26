# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check actual executed keyset statements with both SQL dialect compilers."""

from datetime import datetime

import pytest
from nhx.core.entities.app.repository.pagination import EntityPageToken, entity_page_scope
from sqlalchemy import event
from sqlalchemy.dialects import postgresql, sqlite


@pytest.mark.parametrize("sort", ["created_at", "-created_at", None])
async def test_keyset_query_has_total_order_and_no_offset(entity_repo, session_maker, setup_workspaces, sort):
    effective_sort = "-created_at" if sort is None else sort
    scope = entity_page_scope(
        workspace="workspace-1",
        entity_type="widget",
        sort=effective_sort,
        filter_op=None,
        relationship_child_workspaces=None,
    )
    token = EntityPageToken(v=1, t=datetime(2026, 1, 1), i="widget-boundary", q=scope).encode()
    statements = []
    async with session_maker() as session:
        engine = session.get_bind()

        def capture(conn, clauseelement, multiparams, params, execution_options):
            statements.append(clauseelement)

        event.listen(engine, "before_execute", capture)
        try:
            await entity_repo.list_entities(
                workspace="workspace-1", entity_type="widget", sort=sort, page_token=token, page_size=3, session=session
            )
        finally:
            event.remove(engine, "before_execute", capture)
    assert len(statements) == 2  # Full filtered count, then bounded rows.
    count, query = statements
    assert query._offset_clause is None
    for dialect in [postgresql.dialect(), sqlite.dialect()]:
        sql = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        direction = "DESC" if effective_sort.startswith("-") else "ASC"
        assert f"ORDER BY entities.created_at {direction}, entities.id ASC" in sql
        # SQLite's compiler may emit OFFSET 0 as part of a LIMIT; the SQLAlchemy
        # query itself must not have a numbered-page offset.
        operator = "<" if effective_sort.startswith("-") else ">"
        assert f"entities.created_at {operator}" in sql
        assert "entities.created_at =" in sql
        assert "entities.id > 'widget-boundary'" in sql
        count_sql = str(count.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        assert "widget-boundary" not in count_sql
