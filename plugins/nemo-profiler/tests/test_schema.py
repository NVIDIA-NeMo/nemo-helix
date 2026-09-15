# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for row-schema derivation (from a declared arrow schema and from sampled rows)."""

import pyarrow as pa
from nemo_platform_plugin.files.profile import FeatureSchema
from nemo_profiler_plugin.dataset.schema import MAX_COLUMNS, SchemaFold, arrow_schema_was_capped, derive_features


def _fields(feature: FeatureSchema | None) -> list[FeatureSchema]:
    """The named child fields of a container node.

    Asserting them present rather than indexing through the optionals states the same expectation
    the test already has -- this node is a struct, so it has fields -- and states it as a failure
    the test reports rather than an `AttributeError` from inside a comprehension.
    """
    assert feature is not None, "expected a container node, found none"
    assert feature.fields is not None, f"expected {feature.name!r} to have child fields"
    return feature.fields


# --- from a declared arrow schema (parquet) ------------------------------------------------------


def test_from_arrow_scalars_keep_declared_widths():
    schema = pa.schema(
        [("s", pa.string()), ("i", pa.int64()), ("i32", pa.int32()), ("b", pa.bool_()), ("f", pa.float64())]
    )
    features = {f.name: f.dtype for f in derive_features(schema)}
    assert features == {"s": "string", "i": "int64", "i32": "int32", "b": "bool", "f": "float64"}


def test_from_arrow_list_of_role_content_structs_is_messages():
    schema = pa.schema([("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])))])
    feature = derive_features(schema)[0]
    assert feature.dtype == "messages"
    assert feature.items.dtype == "struct"
    assert [f.name for f in _fields(feature.items)] == ["role", "content"]


def test_from_arrow_fixed_and_variable_lists_agree_on_shape():
    # A fixed-size list is still a list of its element type. The constant length itself is no longer
    # recorded, so the two cases must be indistinguishable rather than one silently losing `items`.
    fixed = derive_features(pa.schema([("embedding", pa.list_(pa.float32(), 768))]))[0]
    assert (fixed.dtype, fixed.items.dtype) == ("list", "float32")

    variable = derive_features(pa.schema([("tags", pa.list_(pa.string()))]))[0]
    assert (variable.dtype, variable.items.dtype) == ("list", "string")


def test_from_arrow_dictionary_encoding_unwraps_to_its_value_type():
    # `DataFrame.to_parquet` writes a dictionary column for every `category` dtype. It is a storage
    # choice, not a logical type -- pyarrow decodes it on the way out -- so it must type as what it
    # holds. Typing it as `json` cost the column its stats, its role, and the classification.
    schema = pa.schema(
        [
            ("label", pa.dictionary(pa.int32(), pa.string())),
            ("code", pa.dictionary(pa.int8(), pa.int64())),
        ]
    )
    assert {f.name: f.dtype for f in derive_features(schema)} == {"label": "string", "code": "int64"}


def test_from_arrow_dictionary_inside_a_list_still_unwraps():
    schema = pa.schema([("tags", pa.list_(pa.dictionary(pa.int32(), pa.string())))])
    feature = derive_features(schema)[0]
    assert (feature.dtype, feature.items.dtype) == ("list", "string")


def test_from_arrow_temporal_and_binary_types_are_named_not_json():
    # None of these carry statistics, but `json` reads as "not understood" and these are understood.
    # Naming them also lets the role gates refuse them deliberately rather than by omission.
    schema = pa.schema(
        [
            ("d", pa.date32()),
            ("ts", pa.timestamp("s")),
            ("t", pa.time64("us")),
            ("b", pa.binary()),
            ("dec", pa.decimal128(10, 2)),
        ]
    )
    assert {f.name: f.dtype for f in derive_features(schema)} == {
        "d": "date",
        "ts": "timestamp",
        "t": "time",
        "b": "binary",
        "dec": "decimal",
    }


# --- inferred from rows, one batch at a time -----------------------------------------------------
#
# Against `SchemaFold` directly, because it is the only implementation. A second one took the rows
# as a list and existed for these tests to call; the two drifted, and the tests pointed at the half
# that never ran.


def _folded(rows, name):
    """One column's schema, folded a row at a time the way the profiler feeds it."""
    fold = SchemaFold(name)
    for row in rows:
        fold.update([row.get(name)])
    return fold.finalize()


def test_from_rows_scalars_widen_int_and_float():
    rows = [{"a": 1, "b": 1.5, "c": "x", "d": True}, {"a": 2, "b": 2, "c": "y", "d": False}]
    assert {name: _folded(rows, name).dtype for name in "abcd"} == {
        "a": "int64",
        "b": "float64",
        "c": "string",
        "d": "bool",
    }


def test_from_rows_list_of_role_content_structs_is_messages():
    rows = [{"conv": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    feature = _folded(rows, "conv")
    assert feature.dtype == "messages"
    assert {f.name for f in _fields(feature.items)} == {"role", "content"}


def test_from_rows_from_value_spelling_is_messages():
    # The same structure is also spelled {from, value}. Recognizing only {role, content} left it a
    # plain list, which then failed the messages dtype gate and profiled as `unknown` with no stats.
    rows = [{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}]
    feature = _folded(rows, "conversations")
    assert feature.dtype == "messages"
    assert {f.name for f in _fields(feature.items)} == {"from", "value"}


def test_from_arrow_from_value_spelling_is_messages():
    schema = pa.schema([("conversations", pa.list_(pa.struct([("from", pa.string()), ("value", pa.string())])))])
    assert derive_features(schema)[0].dtype == "messages"


def test_from_rows_lists_infer_their_element_type():
    constant = _folded([{"e": [0.1, 0.2, 0.3]}, {"e": [0.4, 0.5, 0.6]}], "e")
    assert (constant.dtype, constant.items.dtype) == ("list", "float64")

    variable = _folded([{"e": [1, 2]}, {"e": [1, 2, 3]}], "e")
    assert (variable.dtype, variable.items.dtype) == ("list", "int64")


def test_from_rows_nested_struct():
    feature = _folded([{"meta": {"id": 1, "src": "a"}}, {"meta": {"id": 2, "src": "b"}}], "meta")
    assert feature.dtype == "struct"
    assert {f.name for f in _fields(feature)} == {"id", "src"}


def test_from_rows_all_null_column_is_json():
    assert _folded([{"x": None}, {"x": None}], "x").dtype == "json"


def test_a_declared_schema_reports_truncation_from_the_schema_not_the_result():
    # The result cannot answer this. A file with exactly MAX_COLUMNS columns is described completely
    # and is the same length as one that was cut short, so asking the returned list called the first
    # one broken -- the same failure as leaving the second silent, in the other direction.
    exact = pa.schema([(f"c{i}", pa.int64()) for i in range(MAX_COLUMNS)])
    over = pa.schema([(f"c{i}", pa.int64()) for i in range(MAX_COLUMNS + 1)])
    assert len(derive_features(exact)) == MAX_COLUMNS
    assert arrow_schema_was_capped(exact) is False
    assert arrow_schema_was_capped(over) is True


def test_a_declared_structs_fields_are_bounded_too():
    # The inferred half was closed first and this one was left open, so a parquet carrying a runaway
    # struct kept its whole tree. It is also the only signal a declared partition has: a declared
    # column folds no schema of its own, so `RowFold.columns_were_capped` is structurally False.
    wide = pa.schema([("meta", pa.struct([(f"k{i}", pa.int64()) for i in range(MAX_COLUMNS + 500)]))])
    assert len(_fields(derive_features(wide)[0])) == MAX_COLUMNS
    assert arrow_schema_was_capped(wide) is True

    # And it has to look inside containers, not just at the top level.
    in_list = pa.schema([("a", pa.list_(pa.struct([(f"k{i}", pa.int64()) for i in range(MAX_COLUMNS + 1)])))])
    assert arrow_schema_was_capped(in_list) is True

    narrow = pa.schema([("meta", pa.struct([(f"k{i}", pa.int64()) for i in range(10)]))])
    assert arrow_schema_was_capped(narrow) is False


def test_a_structs_fields_are_bounded_like_a_rows_columns():
    # One level of nesting used to defeat the cap outright: rows carrying a unique key inside `meta`
    # left the top level at one column while minting a fold per row underneath it, and the whole
    # tree went into the stored profile.
    fold = SchemaFold("meta")
    for i in range(MAX_COLUMNS + 500):
        fold.update([{f"k{i}": 1}])
    feature = fold.finalize()
    assert len(_fields(feature)) == MAX_COLUMNS
    assert fold.was_capped() is True


def test_a_struct_that_fits_is_not_reported_as_capped():
    fold = SchemaFold("meta")
    fold.update([{"a": 1, "b": 2}])
    assert fold.was_capped() is False
