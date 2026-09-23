# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for in-memory filter-tree evaluation via ``InMemoryFilterRepository``.

These pin the native-Python contract documented on ``InMemoryFilterRepository``:
values compared by their Python type (no JSON-to-text coercion), absent/None
satisfying only ``$eq None`` (SQL three-valued NULL logic), and ``$like`` as a
literal substring test. A narrower SQL-parity safety net for the cases where
native and SQL semantics agree lives at
services/core/entities/tests/test_filter_matches_sql_parity.py.
"""

from datetime import datetime

import pytest
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.in_memory_filter import InMemoryFilterRepository


class Entity:
    """Simple object whose attributes mirror entity columns (plain + data JSON)."""

    def __init__(self, name=None, score=None, data=None):
        self.name = name
        self.score = score
        self.data = data if data is not None else {}


def cmp(operator, field, value):
    return ComparisonOperation(operator=operator, field=field, value=value)


def evaluate(op, entity):
    """Run ``op`` against ``entity`` in memory — the front door real callers use."""
    return op.apply(InMemoryFilterRepository(entity))


class TestEqPlainAttribute:
    def test_eq_string_hit(self):
        assert evaluate(cmp(FilterOperator.EQ, "name", "llama"), Entity(name="llama")) is True

    def test_eq_string_miss(self):
        assert evaluate(cmp(FilterOperator.EQ, "name", "llama"), Entity(name="other")) is False

    def test_eq_int(self):
        assert evaluate(cmp(FilterOperator.EQ, "score", 5), Entity(score=5)) is True
        assert evaluate(cmp(FilterOperator.EQ, "score", 5), Entity(score=6)) is False

    def test_eq_none_matches_none_attribute(self):
        assert evaluate(cmp(FilterOperator.EQ, "name", None), Entity(name=None)) is True

    def test_eq_none_does_not_match_set_attribute(self):
        assert evaluate(cmp(FilterOperator.EQ, "name", None), Entity(name="x")) is False

    def test_eq_works_on_dict_entity(self):
        assert evaluate(cmp(FilterOperator.EQ, "name", "llama"), {"name": "llama"}) is True


class TestEqDataPath:
    def test_eq_nested_string(self):
        e = Entity(data={"finetuning_type": "LoRA"})
        assert evaluate(cmp(FilterOperator.EQ, "data.finetuning_type", "LoRA"), e) is True
        assert evaluate(cmp(FilterOperator.EQ, "data.finetuning_type", "lora"), e) is False

    def test_eq_int_native_not_text_coerced(self):
        # Native comparison: an int value matches a stored int, but the string
        # "5" does NOT match the integer 5 (no JSON-to-text coercion).
        e = Entity(data={"score": 5})
        assert evaluate(cmp(FilterOperator.EQ, "data.score", 5), e) is True
        assert evaluate(cmp(FilterOperator.EQ, "data.score", "5"), e) is False

    def test_eq_float(self):
        e = Entity(data={"x": 1.5})
        assert evaluate(cmp(FilterOperator.EQ, "data.x", 1.5), e) is True

    def test_eq_bool_true(self):
        assert evaluate(cmp(FilterOperator.EQ, "data.flag", True), Entity(data={"flag": True})) is True
        assert evaluate(cmp(FilterOperator.EQ, "data.flag", True), Entity(data={"flag": False})) is False

    def test_eq_bool_false(self):
        assert evaluate(cmp(FilterOperator.EQ, "data.flag", False), Entity(data={"flag": False})) is True

    def test_eq_none_matches_explicit_null(self):
        assert evaluate(cmp(FilterOperator.EQ, "data.k", None), Entity(data={"k": None})) is True

    def test_eq_none_matches_missing_key(self):
        assert evaluate(cmp(FilterOperator.EQ, "data.k", None), Entity(data={})) is True

    def test_eq_value_does_not_match_missing_key(self):
        assert evaluate(cmp(FilterOperator.EQ, "data.k", "v"), Entity(data={})) is False

    def test_eq_deeply_nested(self):
        e = Entity(data={"a": {"b": {"c": "deep"}}})
        assert evaluate(cmp(FilterOperator.EQ, "data.a.b.c", "deep"), e) is True
        assert evaluate(cmp(FilterOperator.EQ, "data.a.b.c", "shallow"), e) is False

    def test_eq_descend_into_non_dict_is_missing(self):
        e = Entity(data={"a": "scalar"})
        assert evaluate(cmp(FilterOperator.EQ, "data.a.b", None), e) is True


class TestLike:
    def test_like_substring_hit_plain(self):
        assert evaluate(cmp(FilterOperator.LIKE, "name", "lam"), Entity(name="llama")) is True

    def test_like_case_insensitive(self):
        assert evaluate(cmp(FilterOperator.LIKE, "name", "LLAMA"), Entity(name="my-llama-2")) is True

    def test_like_miss(self):
        assert evaluate(cmp(FilterOperator.LIKE, "name", "zebra"), Entity(name="llama")) is False

    def test_like_not_regex(self):
        # `%` is a literal here, NOT a wildcard.
        assert evaluate(cmp(FilterOperator.LIKE, "name", "a%b"), Entity(name="xaybz")) is False
        assert evaluate(cmp(FilterOperator.LIKE, "name", "a%b"), Entity(name="xa%bz")) is True

    def test_like_none_plain_never_matches(self):
        assert evaluate(cmp(FilterOperator.LIKE, "name", "x"), Entity(name=None)) is False

    def test_like_data_path(self):
        e = Entity(data={"desc": "a Llama model"})
        assert evaluate(cmp(FilterOperator.LIKE, "data.desc", "llama"), e) is True

    def test_like_data_absent_never_matches(self):
        # A missing key is absent (not the text "null"); $like never matches it.
        assert evaluate(cmp(FilterOperator.LIKE, "data.k", "ull"), Entity(data={})) is False


class TestInNin:
    def test_in_plain_hit(self):
        assert evaluate(cmp(FilterOperator.IN, "name", ["a", "b"]), Entity(name="b")) is True

    def test_in_plain_miss(self):
        assert evaluate(cmp(FilterOperator.IN, "name", ["a", "b"]), Entity(name="c")) is False

    def test_in_none_plain_never_matches(self):
        assert evaluate(cmp(FilterOperator.IN, "name", ["a"]), Entity(name=None)) is False

    def test_nin_plain_hit(self):
        assert evaluate(cmp(FilterOperator.NIN, "name", ["a", "b"]), Entity(name="c")) is True

    def test_nin_plain_excludes_member(self):
        assert evaluate(cmp(FilterOperator.NIN, "name", ["a", "b"]), Entity(name="a")) is False

    def test_nin_none_plain_never_matches(self):
        # NULL NOT IN (...) is NULL -> not matched.
        assert evaluate(cmp(FilterOperator.NIN, "name", ["a"]), Entity(name=None)) is False

    def test_in_data_native_membership(self):
        # Native membership: an int matches a numeric member, but the string
        # "5" does not match the stored integer 5.
        e = Entity(data={"score": 5})
        assert evaluate(cmp(FilterOperator.IN, "data.score", [5, 6]), e) is True
        assert evaluate(cmp(FilterOperator.IN, "data.score", ["5"]), e) is False

    def test_nin_data_absent_never_matches(self):
        # A missing key is absent, not the text "null": NULL NOT IN (...) is
        # NULL, so $nin does NOT match a missing field (even though "v" isn't it).
        assert evaluate(cmp(FilterOperator.NIN, "data.k", ["v"]), Entity(data={})) is False


class TestOrdered:
    def test_gt_plain_int(self):
        assert evaluate(cmp(FilterOperator.GT, "score", 5), Entity(score=9)) is True
        assert evaluate(cmp(FilterOperator.GT, "score", 5), Entity(score=5)) is False

    def test_gte_plain(self):
        assert evaluate(cmp(FilterOperator.GTE, "score", 5), Entity(score=5)) is True

    def test_lt_plain(self):
        assert evaluate(cmp(FilterOperator.LT, "score", 5), Entity(score=4)) is True

    def test_lte_plain(self):
        assert evaluate(cmp(FilterOperator.LTE, "score", 5), Entity(score=5)) is True

    def test_ordered_none_plain_never_matches(self):
        assert evaluate(cmp(FilterOperator.GT, "score", 5), Entity(score=None)) is False
        assert evaluate(cmp(FilterOperator.LT, "score", 5), Entity(score=None)) is False

    def test_gt_data_numeric(self):
        e = Entity(data={"score": 100})
        assert evaluate(cmp(FilterOperator.GT, "data.score", 9), e) is True

    def test_lt_data_numeric_string_is_incomparable(self):
        # Native: a stored numeric *string* and an int operand are incomparable
        # (no float coercion), so the comparison yields no match.
        e = Entity(data={"score": "9"})
        assert evaluate(cmp(FilterOperator.LT, "data.score", 10), e) is False

    def test_ordered_data_text_compare(self):
        e = Entity(data={"tier": "m"})
        assert evaluate(cmp(FilterOperator.GT, "data.tier", "a"), e) is True
        assert evaluate(cmp(FilterOperator.LT, "data.tier", "z"), e) is True

    def test_gt_data_absent_never_matches(self):
        # Absent field never satisfies an ordered comparison (SQL NULL < x).
        assert evaluate(cmp(FilterOperator.GT, "data.k", -1), Entity(data={})) is False
        assert evaluate(cmp(FilterOperator.GT, "data.k", 0), Entity(data={})) is False

    def test_ordered_datetime_iso_string(self):
        e = Entity(score=datetime(2024, 6, 1, 12, 0, 0))
        assert evaluate(cmp(FilterOperator.GT, "score", "2024-01-01T00:00:00"), e) is True
        assert evaluate(cmp(FilterOperator.LT, "score", "2024-01-01T00:00:00"), e) is False


class TestLogical:
    def test_and_all_true(self):
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                cmp(FilterOperator.EQ, "name", "llama"),
                cmp(FilterOperator.GT, "score", 5),
            ],
        )
        assert evaluate(op, Entity(name="llama", score=9)) is True
        assert evaluate(op, Entity(name="llama", score=1)) is False

    def test_or_any_true(self):
        op = LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                cmp(FilterOperator.EQ, "name", "a"),
                cmp(FilterOperator.EQ, "name", "b"),
            ],
        )
        assert evaluate(op, Entity(name="b")) is True
        assert evaluate(op, Entity(name="c")) is False

    def test_not_negates(self):
        op = LogicalOperation(
            operator=FilterOperator.NOT,
            operations=[cmp(FilterOperator.EQ, "name", "llama")],
        )
        assert evaluate(op, Entity(name="other")) is True
        assert evaluate(op, Entity(name="llama")) is False

    def test_nested_and_or_not(self):
        # name == "llama" AND NOT (score < 5)
        op = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                cmp(FilterOperator.EQ, "name", "llama"),
                LogicalOperation(
                    operator=FilterOperator.NOT,
                    operations=[cmp(FilterOperator.LT, "score", 5)],
                ),
            ],
        )
        assert evaluate(op, Entity(name="llama", score=9)) is True
        assert evaluate(op, Entity(name="llama", score=2)) is False
        assert evaluate(op, Entity(name="other", score=9)) is False

    def test_not_requires_exactly_one_operand(self):
        op = LogicalOperation(
            operator=FilterOperator.NOT,
            operations=[
                cmp(FilterOperator.EQ, "name", "a"),
                cmp(FilterOperator.EQ, "name", "b"),
            ],
        )
        with pytest.raises(ValueError, match="exactly one operand"):
            evaluate(op, Entity(name="a"))


class TestUnsupported:
    def test_exists_raises_not_implemented(self):
        # Use a data.* field so resolution succeeds and the operator dispatch is
        # reached; $exists is relationship-only and must raise NotImplementedError.
        with pytest.raises(NotImplementedError):
            evaluate(cmp(FilterOperator.EXISTS, "data.adapters", True), Entity())

    def test_unknown_field_raises_value_error(self):
        with pytest.raises(ValueError, match="does not exist"):
            evaluate(cmp(FilterOperator.EQ, "nonexistent", "x"), Entity())


class TestNativeSemanticsEdgeCases:
    """Pin the native-Python contract on cases where it intentionally diverges
    from one or both SQL backends (no JSON-to-text coercion, no numeric cast)."""

    def test_eq_bool_via_native_value(self):
        # Booleans compare natively; use $eq for booleans rather than $like/$in,
        # which see the Python value, not a "1"/"0" (SQLite) or "true"/"false"
        # (PostgreSQL) text rendering.
        assert evaluate(cmp(FilterOperator.EQ, "data.flag", True), Entity(data={"flag": True})) is True
        assert evaluate(cmp(FilterOperator.EQ, "data.flag", False), Entity(data={"flag": True})) is False

    def test_like_on_bool_uses_python_str(self):
        # No SQL bool text rendering: str(True) == "True", so "1" is not a substring.
        assert evaluate(cmp(FilterOperator.LIKE, "data.flag", "1"), Entity(data={"flag": True})) is False
        assert evaluate(cmp(FilterOperator.LIKE, "data.flag", "tru"), Entity(data={"flag": True})) is True

    def test_ordered_numeric_against_text_is_incomparable(self):
        # No lenient numeric cast: comparing stored text to an int operand is a
        # type mismatch and yields no match (neither SQLite's 5.0 nor PG's error).
        e = Entity(data={"n": "5abc"})
        assert evaluate(cmp(FilterOperator.GT, "data.n", 4), e) is False
        assert evaluate(cmp(FilterOperator.LT, "data.n", 6), e) is False

    def test_ordered_incomparable_types_returns_false(self):
        # Comparing a str column to an int is incomparable in Python; the
        # repository treats a TypeError as "no match" (SQL's NULL/three-valued result).
        assert evaluate(cmp(FilterOperator.GT, "name", 5), Entity(name="llama")) is False
        assert evaluate(cmp(FilterOperator.LT, "name", 5), Entity(name="llama")) is False

    def test_comparison_op_with_logical_operator_raises(self):
        # A ComparisonOperation carrying a logical operator is malformed; apply()
        # rejects it rather than silently mis-evaluating.
        op = ComparisonOperation(operator=FilterOperator.AND, field="name", value="x")
        with pytest.raises(ValueError, match="Unknown comparison operator"):
            evaluate(op, Entity(name="x"))

    def test_logical_op_with_comparison_operator_raises(self):
        op = LogicalOperation(
            operator=FilterOperator.EQ,
            operations=[cmp(FilterOperator.EQ, "name", "x")],
        )
        with pytest.raises(ValueError, match="Unknown logical operator"):
            evaluate(op, Entity(name="x"))
