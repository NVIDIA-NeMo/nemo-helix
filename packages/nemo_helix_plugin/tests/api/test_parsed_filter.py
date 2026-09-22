# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical tests for ParsedFilter — extract, remove, has, and_with, to_response,
and the make_filter_dep FastAPI integration. Imports go through the canonical
``nemo_helix_plugin.api.*`` paths.

Entity-aware behavior (translate_operation, bool coercion, _get_entity_field_map)
is tested in ``packages/nhx_common/tests/api/test_parsed_filter.py`` since those
helpers live on the entity-aware ``nhx.common.entities.values.Filter`` subclass.
"""

from fastapi import Depends, FastAPI
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.schema import Filter
from starlette.testclient import TestClient


class SampleFilter(Filter):
    status: str | None = None
    name: str | None = None


def _make_app(filter_model):
    app = FastAPI()
    FilterDep = make_filter_dep(filter_model)

    @app.get("/items")
    async def list_items(parsed: ParsedFilter = Depends(FilterDep)):
        return {
            "operation": parsed.operation.to_dict() if parsed.operation else None,
            "extracted_status": parsed.extract("status"),
        }

    return app


class TestParsedFilterExtract:
    def test_extract_eq_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf = ParsedFilter(operation=op)
        assert pf.extract("status") == "active"

    def test_extract_from_and(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active"),
                ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
            ],
        )
        pf = ParsedFilter(operation=op)
        assert pf.extract("status") == "active"
        assert pf.extract("name") is None  # $like, not $eq

    def test_extract_from_none_operation(self):
        pf = ParsedFilter(operation=None)
        assert pf.extract("status") is None


class TestParsedFilterRemove:
    def test_remove_single_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf = ParsedFilter(operation=op)
        val = pf.remove("status")
        assert val == "active"
        assert pf.operation is None

    def test_remove_from_and(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active"),
                ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
            ],
        )
        pf = ParsedFilter(operation=op)
        val = pf.remove("status")
        assert val == "active"
        assert isinstance(pf.operation, ComparisonOperation)
        assert pf.operation.field == "name"

    def test_remove_non_eq_not_removed(self):
        op = ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama")
        pf = ParsedFilter(operation=op)
        val = pf.remove("name")
        assert val is None
        assert pf.operation is not None


class TestParsedFilterHas:
    def test_has_none_returns_false(self):
        pf = ParsedFilter(operation=None)
        assert pf.has("status") is False

    def test_has_top_level_non_eq(self):
        pf = ParsedFilter(operation=ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="foo%"))
        assert pf.has("name") is True

    def test_has_deeply_nested(self):
        pf = ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
                    LogicalOperation(
                        operator=FilterOperator.OR,
                        operations=[
                            ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a"),
                            ComparisonOperation(operator=FilterOperator.EQ, field="project", value="p"),
                        ],
                    ),
                ],
            )
        )
        assert pf.has("status") is True
        assert pf.has("project") is True

    def test_has_returns_false_for_absent_field(self):
        pf = ParsedFilter(operation=ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"))
        assert pf.has("status") is False


class TestParsedFilterAndWith:
    def test_and_with_none_sets_extra(self):
        pf = ParsedFilter(operation=None)
        extra = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf.and_with(extra)
        assert pf.operation is extra

    def test_and_with_existing_and_appends(self):
        existing = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active"),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
            ],
        )
        pf = ParsedFilter(operation=existing)
        extra = ComparisonOperation(operator=FilterOperator.EQ, field="created_at", value="2026-01-01")
        pf.and_with(extra)
        assert isinstance(pf.operation, LogicalOperation)
        assert len(pf.operation.operations) == 3
        assert pf.operation.operations[-1] is extra


class TestParsedFilterToResponse:
    def test_with_operation_no_field_map(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf = ParsedFilter(operation=op)
        assert pf.to_response() == {"status": {"$eq": "active"}}

    def test_empty(self):
        pf = ParsedFilter(operation=None)
        assert pf.to_response() is None


class TestMakeFilterDep:
    def test_bracket_eq(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[status]": "active"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation"] is not None
        assert data["extracted_status"] == "active"

    def test_bracket_like(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[name][$like]": "llama"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$like": "llama"}}

    def test_json_syntax(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get('/items?filter={"status":"active"}')
        assert resp.status_code == 200
        assert resp.json()["extracted_status"] == "active"

    def test_text_syntax(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter": 'name~"llama"'})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$like": "llama"}}

    def test_no_filter(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items")
        assert resp.status_code == 200
        assert resp.json()["operation"] is None

    def test_unknown_field_400(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[bogus]": "value"})
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]

    def test_full_tree_with_multiple_fields(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[status]": "active", "filter[name][$like]": "llama"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert "$and" in op
        assert len(op["$and"]) == 2

    def test_mixed_logical_root_with_sibling_rejected(self):
        """``{"$or":[...], "name":"x"}`` is ambiguous — historically the parser
        short-circuited on the logical operator and silently dropped the
        sibling, producing overbroad results. Now rejected at the boundary."""
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter": '{"$or":[{"status":"a"}],"name":"foo"}'})
        assert resp.status_code == 400
        assert "logical operator" in resp.json()["detail"]

    def test_mixed_logical_root_with_explicit_and_passes(self):
        """The disambiguating shape — wrap siblings under explicit ``$and``."""
        client = TestClient(_make_app(SampleFilter))
        resp = client.get(
            "/items",
            params={"filter": '{"$and":[{"$or":[{"status":"a"}]},{"name":"foo"}]}'},
        )
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert "$and" in op
        clauses = op["$and"]
        assert any("$or" in c for c in clauses)
        assert any("name" in c for c in clauses)

    def test_field_level_logical_with_sibling_op_rejected(self):
        """Same ambiguity at the field level: ``{"$or":[...], "$gte":...}``."""
        client = TestClient(_make_app(SampleFilter))
        resp = client.get(
            "/items",
            params={"filter": '{"created_at":{"$or":[{"$gte":"2024-01-01"}],"$lte":"2025-01-01"}}'},
        )
        assert resp.status_code == 400
        assert "logical operator" in resp.json()["detail"]
