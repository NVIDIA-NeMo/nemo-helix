# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ParsedFilter — extract, remove, and make_filter_dep."""

from typing import Annotated, Optional, Union

from fastapi import Depends, FastAPI
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.entities.values import DatetimeFilter, Filter, StringFilter, map_entity_field
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Test filter model
# ---------------------------------------------------------------------------


class SampleFilter(Filter):
    status: str | None = None
    name: str | None = None
    created_at: DatetimeFilter | None = None


class BoolCoercibleFilter(Filter):
    """Filter with a bool-coercible field (Union[..., bool, str]) and a pure bool field."""

    base_model: Optional[Union[bool, str]] = None
    pure_bool: Optional[bool] = None
    name: str | None = None


class StringFieldFilter(Filter):
    """Filter with a StringFilter-typed field to exercise operator bracket notation."""

    name: StringFilter | str | None = None


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


# ---------------------------------------------------------------------------
# ParsedFilter.extract
# ---------------------------------------------------------------------------


class TestParsedFilterExtract:
    def test_extract_eq_field(self):
        # In real usage, operations are already translated (data.status not status)
        op = ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        assert pf.extract("status") == "active"

    def test_extract_base_field(self):
        # Base fields (name) don't get data. prefix
        op = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="llama")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        assert pf.extract("name") == "llama"

    def test_extract_from_and(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active"),
                ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
            ],
        )
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        assert pf.extract("status") == "active"
        assert pf.extract("name") is None  # $like, not $eq

    def test_extract_missing_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        assert pf.extract("name") is None

    def test_extract_from_none_operation(self):
        pf = ParsedFilter(operation=None, _field_map=SampleFilter._get_entity_field_map())
        assert pf.extract("status") is None


# ---------------------------------------------------------------------------
# ParsedFilter.remove
# ---------------------------------------------------------------------------


class TestParsedFilterRemove:
    def test_remove_single_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        val = pf.remove("status")
        assert val == "active"
        assert pf.operation is None

    def test_remove_from_and(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active"),
                ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
            ],
        )
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        val = pf.remove("status")
        assert val == "active"
        assert isinstance(pf.operation, ComparisonOperation)
        assert pf.operation.field == "name"

    def test_remove_base_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="llama")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        val = pf.remove("name")
        assert val == "llama"
        assert pf.operation is None

    def test_remove_missing_field(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="active")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        val = pf.remove("name")
        assert val is None
        assert pf.operation is not None

    def test_remove_from_none(self):
        pf = ParsedFilter(operation=None, _field_map=SampleFilter._get_entity_field_map())
        result = pf.remove("status")
        assert result is None

    def test_remove_non_eq_not_removed(self):
        op = ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        val = pf.remove("name")
        assert val is None
        assert pf.operation is not None


# ---------------------------------------------------------------------------
# ParsedFilter.to_response
# ---------------------------------------------------------------------------


class TestParsedFilterHas:
    """``has`` walks the entire tree (unlike ``extract``/``remove``) and
    accepts both user-facing and entity-mapped names so the caller doesn't
    have to know which form the tree happens to be in."""

    def test_has_none_returns_false(self):
        pf = ParsedFilter(operation=None)
        assert pf.has("status") is False

    def test_has_top_level_eq(self):
        pf = ParsedFilter(operation=ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a"))
        assert pf.has("status") is True

    def test_has_top_level_non_eq(self):
        # Operator-agnostic — different from extract/remove, which only see $eq.
        pf = ParsedFilter(operation=ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="foo%"))
        assert pf.has("name") is True

    def test_has_nested_in_or(self):
        pf = ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a"),
                    ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
                ],
            )
        )
        assert pf.has("status") is True
        assert pf.has("name") is True

    def test_has_nested_in_not(self):
        pf = ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a")],
            )
        )
        assert pf.has("status") is True

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

    def test_has_does_not_match_sibling_field(self):
        pf = ParsedFilter(
            operation=LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
                    ComparisonOperation(operator=FilterOperator.EQ, field="project", value="p"),
                ],
            )
        )
        assert pf.has("status") is False

    def test_has_resolves_user_facing_name_to_mapped(self):
        """Caller passes ``status`` (user-facing); tree contains ``data.status`` (post-translate)."""
        pf = ParsedFilter(
            operation=ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="a"),
            _field_map={"status": "data.status"},
        )
        assert pf.has("status") is True

    def test_has_resolves_mapped_name_when_tree_is_untranslated(self):
        """Caller passes ``status``; tree still has the un-translated ``status`` (no translate_operation pass yet)."""
        pf = ParsedFilter(
            operation=ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a"),
            _field_map={"status": "data.status"},
        )
        assert pf.has("status") is True

    def test_has_accepts_mapped_name_directly(self):
        """Caller passes the entity-mapped name directly; should still match a tree containing it."""
        pf = ParsedFilter(
            operation=ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="a"),
            _field_map={"status": "data.status"},
        )
        assert pf.has("data.status") is True


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
        assert pf.operation.operator == FilterOperator.AND
        assert len(pf.operation.operations) == 3
        assert pf.operation.operations[-1] is extra

    def test_and_with_non_and_wraps(self):
        existing = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf = ParsedFilter(operation=existing)
        extra = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo")
        pf.and_with(extra)
        assert isinstance(pf.operation, LogicalOperation)
        assert pf.operation.operator == FilterOperator.AND
        assert pf.operation.operations == [existing, extra]

    def test_and_with_or_wraps_in_new_and(self):
        existing = LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="status", value="a"),
                ComparisonOperation(operator=FilterOperator.EQ, field="status", value="b"),
            ],
        )
        pf = ParsedFilter(operation=existing)
        extra = ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo")
        pf.and_with(extra)
        assert isinstance(pf.operation, LogicalOperation)
        assert pf.operation.operator == FilterOperator.AND
        assert pf.operation.operations == [existing, extra]


class TestParsedFilterToResponse:
    def test_with_operation(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="status", value="active")
        pf = ParsedFilter(operation=op, _field_map=SampleFilter._get_entity_field_map())
        assert pf.to_response() == {"status": {"$eq": "active"}}

    def test_empty(self):
        pf = ParsedFilter(operation=None, _field_map=SampleFilter._get_entity_field_map())
        assert pf.to_response() is None


# ---------------------------------------------------------------------------
# make_filter_dep integration tests
# ---------------------------------------------------------------------------


class TestMakeFilterDep:
    def test_bracket_eq(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[status]": "active"})
        assert resp.status_code == 200
        data = resp.json()
        # Bare bracket → $eq, translated to data.status
        assert data["operation"] is not None
        assert data["extracted_status"] == "active"

    def test_bracket_like(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[name][$like]": "llama"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation"] is not None
        assert data["extracted_status"] is None  # no status in query

    def test_json_syntax(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get('/items?filter={"status":"active"}')
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted_status"] == "active"

    def test_no_filter(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items")
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation"] is None

    def test_unknown_field_400(self):
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[bogus]": "value"})
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]

    def test_operation_is_full_tree(self):
        """The operation should contain ALL fields — nothing extracted away."""
        client = TestClient(_make_app(SampleFilter))
        resp = client.get("/items", params={"filter[status]": "active", "filter[name][$like]": "llama"})
        assert resp.status_code == 200
        data = resp.json()
        op = data["operation"]
        # Should be an $and with both fields
        assert "$and" in op
        assert len(op["$and"]) == 2


# ---------------------------------------------------------------------------
# Bool coercion on Union[..., bool, str] fields
# ---------------------------------------------------------------------------


class TestBoolCoercion:
    """Filter fields typed as Union[..., bool, str] should coerce "true"/"false"
    to null/not-null existence checks during translate_operation."""

    def test_false_string_coerced_to_eq_null(self):
        """filter[base_model]=false  → data.base_model $eq null."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value="false")
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, ComparisonOperation)
        assert translated.field == "data.base_model"
        assert translated.operator == FilterOperator.EQ
        assert translated.value is None

    def test_true_string_coerced_to_not_eq_null(self):
        """filter[base_model]=true  → $not { data.base_model $eq null }."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value="true")
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, LogicalOperation)
        assert translated.operator == FilterOperator.NOT
        inner = translated.operations[0]
        assert isinstance(inner, ComparisonOperation)
        assert inner.field == "data.base_model"
        assert inner.operator == FilterOperator.EQ
        assert inner.value is None

    def test_bool_false_coerced_to_eq_null(self):
        """Python bool False → data.base_model $eq null (same as string)."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value=False)
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, ComparisonOperation)
        assert translated.field == "data.base_model"
        assert translated.operator == FilterOperator.EQ
        assert translated.value is None

    def test_bool_true_coerced_to_not_eq_null(self):
        """Python bool True → $not { data.base_model $eq null }."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value=True)
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, LogicalOperation)
        assert translated.operator == FilterOperator.NOT

    def test_string_value_not_coerced(self):
        """A normal string value like 'llama-3' should NOT be coerced."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value="llama-3")
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, ComparisonOperation)
        assert translated.field == "data.base_model"
        assert translated.value == "llama-3"

    def test_pure_bool_field_not_coerced(self):
        """Optional[bool] fields (like 'pure_bool') should NOT get existence coercion."""
        op = ComparisonOperation(operator=FilterOperator.EQ, field="pure_bool", value="true")
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, ComparisonOperation)
        assert translated.field == "data.pure_bool"
        assert translated.value == "true"

    def test_like_operator_not_coerced(self):
        """Only $eq operations should be coerced, not $like."""
        op = ComparisonOperation(operator=FilterOperator.LIKE, field="base_model", value="true")
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, ComparisonOperation)
        assert translated.field == "data.base_model"
        assert translated.operator == FilterOperator.LIKE
        assert translated.value == "true"

    def test_coercion_inside_and(self):
        """Bool coercion works when the field is inside an $and."""
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="base_model", value="false"),
                ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="llama"),
            ],
        )
        translated = BoolCoercibleFilter.translate_operation(op)
        assert isinstance(translated, LogicalOperation)
        assert translated.operator == FilterOperator.AND
        # First operand: base_model=false → $eq null
        bm = translated.operations[0]
        assert isinstance(bm, ComparisonOperation)
        assert bm.field == "data.base_model"
        assert bm.value is None
        # Second operand: name $like unchanged
        nm = translated.operations[1]
        assert isinstance(nm, ComparisonOperation)
        assert nm.field == "name"
        assert nm.value == "llama"


def _bool_coercion_client() -> TestClient:
    """TestClient wired to BoolCoercibleFilter via make_filter_dep."""
    return TestClient(_make_app(BoolCoercibleFilter))


class TestBoolCoercionEndToEnd:
    """End-to-end tests through make_filter_dep with bracket notation."""

    def test_bracket_false_produces_eq_null(self):
        client = _bool_coercion_client()
        resp = client.get("/items", params={"filter[base_model]": "false"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"data.base_model": {"$eq": None}}

    def test_bracket_true_produces_not_eq_null(self):
        client = _bool_coercion_client()
        resp = client.get("/items", params={"filter[base_model]": "true"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"$not": {"data.base_model": {"$eq": None}}}

    def test_bracket_string_value_unchanged(self):
        client = _bool_coercion_client()
        resp = client.get("/items", params={"filter[base_model]": "llama-3"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"data.base_model": {"$eq": "llama-3"}}


# ---------------------------------------------------------------------------
# Namespace field mapping — map_entity_field(..., namespace=True)
# ---------------------------------------------------------------------------


class NamespaceFilter(Filter):
    """Filter exercising a namespace mapping: ``labels.*`` → ``data.labels.*``.

    The ``labels`` knowledge lives entirely here, in the model — the generic
    parser only learns "this field is a namespace" via the annotation.
    """

    name: str | None = None
    labels: Annotated[dict[str, str] | None, map_entity_field("data.labels", namespace=True)] = None


class TestNamespaceTranslate:
    """Prefix rewriting in translate_operation (no per-service knowledge in the core)."""

    def test_subpath_translates(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="labels.eval_category", value="agentic")
        assert NamespaceFilter.translate_operation(op).to_dict() == {"data.labels.eval_category": {"$eq": "agentic"}}

    def test_bare_namespace_translates(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="labels", value="x")
        assert NamespaceFilter.translate_operation(op).to_dict() == {"data.labels": {"$eq": "x"}}

    def test_deep_subpath_translates(self):
        op = ComparisonOperation(operator=FilterOperator.LIKE, field="labels.team.name", value="eval")
        assert NamespaceFilter.translate_operation(op).to_dict() == {"data.labels.team.name": {"$like": "eval"}}

    def test_already_canonical_unchanged(self):
        op = ComparisonOperation(operator=FilterOperator.EQ, field="data.labels.x", value="y")
        assert NamespaceFilter.translate_operation(op).to_dict() == {"data.labels.x": {"$eq": "y"}}

    def test_scalar_field_subpath_not_rewritten(self):
        # ``name`` is a scalar field, not a namespace — its sub-path is left alone
        # here (and rejected at validation time; see TestNamespaceEndToEnd).
        op = ComparisonOperation(operator=FilterOperator.EQ, field="name.foo", value="y")
        assert NamespaceFilter.translate_operation(op).to_dict() == {"name.foo": {"$eq": "y"}}

    def test_namespace_inside_logical_tree(self):
        op = LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="labels.a", value="1"),
                ComparisonOperation(operator=FilterOperator.EQ, field="data.labels.b", value="2"),
            ],
        )
        assert NamespaceFilter.translate_operation(op).to_dict() == {
            "$or": [{"data.labels.a": {"$eq": "1"}}, {"data.labels.b": {"$eq": "2"}}]
        }


class TestNamespaceEndToEnd:
    """End-to-end through make_filter_dep: validation + translation together."""

    def _client(self) -> TestClient:
        return TestClient(_make_app(NamespaceFilter))

    def test_subpath_validates_and_translates(self):
        resp = self._client().get("/items", params={"filter[labels.eval_category]": "agentic"})
        assert resp.status_code == 200
        assert resp.json()["operation"] == {"data.labels.eval_category": {"$eq": "agentic"}}

    def test_data_prefixed_form_still_works(self):
        # Regression: the explicit data.labels.* path must keep working unchanged.
        resp = self._client().get("/items", params={"filter[data.labels.eval_category]": "agentic"})
        assert resp.status_code == 200
        assert resp.json()["operation"] == {"data.labels.eval_category": {"$eq": "agentic"}}

    def test_json_subpath_form(self):
        resp = self._client().get('/items?filter={"labels.team":"eval"}')
        assert resp.status_code == 200
        assert resp.json()["operation"] == {"data.labels.team": {"$eq": "eval"}}

    def test_subpath_in_or_tree(self):
        resp = self._client().get("/items", params={"filter": '{"$or":[{"labels.a":"1"},{"name":"x"}]}'})
        assert resp.status_code == 200
        assert resp.json()["operation"] == {"$or": [{"data.labels.a": {"$eq": "1"}}, {"name": {"$eq": "x"}}]}

    def test_unknown_dotted_field_still_400(self):
        # A non-namespace dotted path must still be rejected — no permissive auto-map.
        resp = self._client().get("/items", params={"filter[foo.bar]": "x"})
        assert resp.status_code == 400
        assert "foo.bar" in resp.json()["detail"]

    def test_scalar_field_subpath_400(self):
        # ``name`` is a scalar field; ``name.foo`` is not a valid namespace sub-path.
        resp = self._client().get("/items", params={"filter[name.foo]": "x"})
        assert resp.status_code == 400


class TestStringFilterEndToEnd:
    """End-to-end tests through make_filter_dep with StringFilter operator brackets.

    ``make_filter_dep`` parses bracket notation into FilterOperation objects
    independently of the field's declared StringFilter type, so widening the
    field to ``StringFilter | str | None`` is purely additive for parsing.
    """

    def _client(self) -> TestClient:
        return TestClient(_make_app(StringFieldFilter))

    def test_bracket_like(self):
        client = self._client()
        resp = client.get("/items", params={"filter[name][$like]": "llama"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$like": "llama"}}

    def test_bracket_in_comma_split(self):
        client = self._client()
        resp = client.get("/items", params={"filter[name][$in]": "a,b,c"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$in": ["a", "b", "c"]}}

    def test_bracket_nin_comma_split(self):
        client = self._client()
        resp = client.get("/items", params={"filter[name][$nin]": "x,y"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$nin": ["x", "y"]}}

    def test_bracket_implicit_eq(self):
        client = self._client()
        resp = client.get("/items", params={"filter[name]": "llama-3"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$eq": "llama-3"}}

    def test_explicit_eq(self):
        client = self._client()
        resp = client.get("/items", params={"filter[name][$eq]": "llama-3"})
        assert resp.status_code == 200
        op = resp.json()["operation"]
        assert op == {"name": {"$eq": "llama-3"}}
