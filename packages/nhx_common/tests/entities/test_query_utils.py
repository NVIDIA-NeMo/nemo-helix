# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for entity store query utilities.

Validates coerce_existence_operator — the utility that translates boolean
filter values on nullable (non-boolean) fields into the correct entity-store
query operators.

See issue #3976 for context on why this utility exists.
"""

import pytest
from nhx.common.entities.query_utils import coerce_existence_operator


class TestCoerceExistenceOperator:
    """Test coerce_existence_operator for all input types."""

    def test_false_returns_eq_null(self):
        """False means 'field is absent or null' → {"$eq": None}."""
        assert coerce_existence_operator(False) == {"$eq": None}

    def test_true_returns_not_eq_null(self):
        """True means 'field is present and non-null' → {"$not": {"$eq": None}}."""
        assert coerce_existence_operator(True) == {"$not": {"$eq": None}}

    def test_string_passes_through(self):
        """Non-boolean values pass through unchanged for direct matching."""
        assert coerce_existence_operator("sft") == "sft"

    def test_none_passes_through(self):
        """None passes through unchanged — caller may have their own semantics for None."""
        assert coerce_existence_operator(None) is None

    def test_dict_passes_through(self):
        """Complex values like dicts pass through unchanged."""
        value = {"template": "You are a helpful assistant."}
        assert coerce_existence_operator(value) == value

    def test_int_passes_through(self):
        """Numeric values pass through unchanged — only bool triggers coercion."""
        assert coerce_existence_operator(42) == 42

    @pytest.mark.parametrize("value", [0, "", [], {}])
    def test_falsy_non_bool_values_pass_through(self, value):
        """Falsy non-boolean values must NOT be coerced — only actual bool triggers it.

        This is critical: Python's isinstance(0, bool) is False (int 0 is not bool),
        so these should all pass through unchanged.
        """
        assert coerce_existence_operator(value) == value
