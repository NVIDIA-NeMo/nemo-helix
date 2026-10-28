# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical tests for ParsedFilter — extract, remove, has, and_with, to_response,
and the make_filter_dep FastAPI integration. Imports go through the canonical
``nemo_helix_plugin.api.*`` paths.

Entity-aware behavior (translate_operation, bool coercion, _get_entity_field_map)
is tested in ``packages/nhx_common/tests/api/test_parsed_filter.py`` since those
helpers live on the entity-aware ``nhx.common.entities.values.Filter`` subclass.
"""

import pytest
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


class LabelsFilter(Filter):
    name: str | None = None
    labels: dict[str, str] | None = None

    @classmethod
    def _get_entity_namespace_map(cls) -> dict[str, str]:
        return {"labels": "data.labels"}


class TestNamespaceDictExpansion:
    """A dict-valued filter on a namespace field means "contains these pairs", not "equals this object"."""

    def _get(self, *args, **kwargs):
        return TestClient(_make_app(LabelsFilter)).get(*args, **kwargs)

    def test_bracket_form_expands_to_subpath(self):
        resp = self._get("/items", params={"filter[labels][team]": "x"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"labels.team": {"$eq": "x"}}

    def test_json_form_expands_to_subpath(self):
        resp = self._get("/items", params={"filter": '{"labels": {"team": "x"}}'})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"labels.team": {"$eq": "x"}}

    def test_multiple_keys_are_anded(self):
        resp = self._get("/items", params={"filter[labels][team]": "x", "filter[labels][tier]": "1"})
        assert resp.status_code == 200, resp.text
        assert sorted(resp.json()["operation"]["$and"], key=str) == [
            {"labels.team": {"$eq": "x"}},
            {"labels.tier": {"$eq": "1"}},
        ]

    def test_bracket_nested_operator_applies_to_key(self):
        resp = self._get("/items", params={"filter[labels][team][in]": "a,b"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"labels.team": {"$in": ["a", "b"]}}

    def test_json_nested_operator_applies_to_key(self):
        resp = self._get("/items", params={"filter": '{"labels": {"team": {"$in": ["a", "b"]}, "tier": "1"}}'})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {
            "$and": [{"labels.team": {"$in": ["a", "b"]}}, {"labels.tier": {"$eq": "1"}}]
        }

    def test_nested_objects_flatten_to_dotted_paths(self):
        resp = self._get("/items", params={"filter[labels][team][lead]": "x"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"labels.team.lead": {"$eq": "x"}}

    def test_expands_inside_logical_tree(self):
        resp = self._get("/items", params={"filter": '{"$or": [{"labels": {"team": "x"}}, {"name": "n"}]}'})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"$or": [{"labels.team": {"$eq": "x"}}, {"name": {"$eq": "n"}}]}

    @pytest.mark.parametrize(
        "params",
        [
            {"filter[labels]": "x"},
            {"filter[labels][$hasKey]": "team"},
            {"filter[labels][$exists]": "true"},
            {"filter[labels][$like]": "x"},
            {"filter": '{"$not": {"labels": {"$hasKey": "team"}}}'},
        ],
    )
    def test_other_bare_namespace_comparisons_rejected(self, params):
        resp = self._get("/items", params=params)
        assert resp.status_code == 400
        assert "key/value pairs" in resp.json()["detail"]

    def test_non_namespace_dict_value_is_not_expanded(self):
        # Only namespace fields get per-key expansion; an object elsewhere is rejected (see TestObjectValues).
        resp = self._get("/items", params={"filter": '{"name": {"first": "x"}}'})
        assert resp.status_code == 400
        assert "not an object" in resp.json()["detail"]

    def test_mixed_operator_and_subkey_rejected(self):
        resp = self._get("/items", params={"filter": '{"labels": {"team": {"$eq": "x", "lead": "y"}}}'})
        assert resp.status_code == 400
        assert "mixes operators" in resp.json()["detail"]

    def test_empty_object_rejected(self):
        resp = self._get("/items", params={"filter": '{"labels": {}}'})
        assert resp.status_code == 400
        assert "key/value pairs" in resp.json()["detail"]


class TestObjectValues:
    """Comparing a non-namespace field to an object matched nothing or crashed the store; now a 400."""

    @pytest.mark.parametrize(
        ("params", "detail"),
        [
            ({"filter": '{"name": {"first": "x"}}'}, "not an object"),
            ({"filter[name][first]": "x"}, "not an object"),
            ({"filter": '{"name": {}}'}, "not an object"),
            ({"filter": '{"name": {"$eq": {"first": "x"}}}'}, "not an object"),
            ({"filter": '{"$or": [{"name": {"first": "x"}}, {"status": "a"}]}'}, "not an object"),
            ({"filter": '{"created_at": {"gte": "2024-01-01"}}'}, "without '$'"),
            ({"filter": '{"name": {"$in": [{"first": "x"}]}}'}, "not objects"),
            ({"filter": '{"name": {"$nin": ["a", {"first": "x"}]}}'}, "not objects"),
            ({"filter": '{"name": {"$eq": "x", "tier": "gold"}}'}, "mixes operators with sub-keys"),
            (
                {"filter[created_at][gte]": "2024-01-01", "filter[created_at][foo]": "x"},
                "mixes operators with sub-keys",
            ),
        ],
    )
    def test_rejected(self, params, detail):
        resp = TestClient(_make_app(SampleFilter)).get("/items", params=params)
        assert resp.status_code == 400
        assert detail in resp.json()["detail"]

    def test_unprefixed_operator_hint_names_the_fix(self):
        resp = TestClient(_make_app(SampleFilter)).get(
            "/items", params={"filter": '{"created_at": {"gte": "2024-01-01"}}'}
        )
        assert "$gte" in resp.json()["detail"]

    def test_elem_match_criteria_object_allowed(self):
        resp = TestClient(_make_app(SampleFilter)).get(
            "/items", params={"filter": '{"name": {"$elemMatch": {"key": "a", "value": "b"}}}'}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"name": {"$elemMatch": {"key": "a", "value": "b"}}}

    def test_scalar_list_still_works(self):
        resp = TestClient(_make_app(SampleFilter)).get("/items", params={"filter": '{"name": {"$in": ["a", "b"]}}'})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"name": {"$in": ["a", "b"]}}

    def test_bracket_operator_alias_still_works(self):
        resp = TestClient(_make_app(SampleFilter)).get("/items", params={"filter[created_at][gte]": "2024-01-01"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["operation"] == {"created_at": {"$gte": "2024-01-01"}}
