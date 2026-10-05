# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the _match_single_value function in nhx.common.api.utils."""

from nhx.common.api.utils import _match_single_value


class TestMatchSingleValue:
    """Test cases for the _match_single_value function."""

    def test_strict_mode_exact_match(self):
        """Test strict mode with exact matches."""
        assert _match_single_value("name", "test", "test", strict=True) is True
        assert _match_single_value("name", 123, 123, strict=True) is True
        assert _match_single_value("name", True, True, strict=True) is True
        assert _match_single_value("name", None, None, strict=True) is True

    def test_strict_mode_no_match(self):
        """Test strict mode with non-matching values."""
        assert _match_single_value("name", "test", "other", strict=True) is False
        assert _match_single_value("name", 123, 456, strict=True) is False
        assert _match_single_value("name", True, False, strict=True) is False
        assert _match_single_value("name", "test", None, strict=True) is False
        assert _match_single_value("name", None, "test", strict=True) is False

    def test_dict_item_value_with_string_filter(self):
        """Test when item_value is dict and filter_value is string."""
        # Should match when dict has 'name' key with matching value
        item = {"name": "test_name", "id": 123}
        assert _match_single_value("name", item, "test_name", strict=True) is True
        assert _match_single_value("name", item, "test_name", strict=False) is True

        # Should not match when dict has 'name' key with different value
        assert _match_single_value("name", item, "other_name", strict=True) is False
        assert _match_single_value("name", item, "other_name", strict=False) is False

        # Should not match when dict doesn't have 'name' key
        item_no_name = {"id": 123, "value": "test"}
        assert _match_single_value("name", item_no_name, "test", strict=True) is False
        assert _match_single_value("name", item_no_name, "test", strict=False) is False

    def test_non_strict_string_matching(self):
        """Test non-strict mode with string values (case-insensitive substring matching)."""
        assert _match_single_value("name", "Hello World", "hello", strict=False) is True
        assert _match_single_value("name", "Hello World", "WORLD", strict=False) is True
        assert _match_single_value("name", "Hello World", "lo Wo", strict=False) is True
        assert _match_single_value("name", "Hello World", "xyz", strict=False) is False

    def test_non_strict_string_item_non_string_filter_fixed(self):
        """Test non-strict mode when item_value is string but filter_value is not string - now works correctly."""
        # After the bug fix, these should return False since non-string filter_value
        # doesn't match string item_value in non-strict mode
        assert _match_single_value("name", "123456", 123, strict=False) is False
        assert _match_single_value("name", "test123test", 123, strict=False) is False
        assert _match_single_value("name", "0", 0, strict=False) is False

    def test_non_strict_string_item_none_filter_fixed(self):
        """Test non-strict mode with string item_value and None filter_value - now works correctly."""
        # After the bug fix, this should return False since None != "test"
        assert _match_single_value("name", "test", None, strict=False) is False

    def test_non_strict_non_string_matching(self):
        """Test non-strict mode with non-string values."""
        # Should fall back to equality check
        assert _match_single_value("name", 123, 123, strict=False) is True
        assert _match_single_value("name", 123, 456, strict=False) is False
        assert _match_single_value("name", True, True, strict=False) is True
        assert _match_single_value("name", True, False, strict=False) is False

    def test_non_strict_none_item_value(self):
        """Test non-strict mode when item_value is None."""
        # None item_value always returns False in non-strict mode (goes to specific None branch)
        assert _match_single_value("name", None, "test", strict=False) is False
        assert _match_single_value("name", None, 123, strict=False) is False
        assert _match_single_value("name", None, True, strict=False) is False
        # Even when both are None in non-strict mode, it returns False
        assert _match_single_value("name", None, None, strict=False) is False

    def test_edge_cases_empty_strings(self):
        """Test various edge cases with empty strings."""
        assert _match_single_value("name", "", "", strict=True) is True
        assert _match_single_value("name", "", "", strict=False) is True
        assert _match_single_value("name", "test", "", strict=False) is True  # Empty string is substring
        assert _match_single_value("name", "", "test", strict=False) is False

    def test_edge_cases_zero_values(self):
        """Test edge cases with zero values."""
        assert _match_single_value("name", 0, 0, strict=True) is True
        assert _match_single_value("name", 0, 0, strict=False) is True

    def test_edge_cases_boolean_values(self):
        """Test edge cases with boolean values."""
        assert _match_single_value("name", False, False, strict=True) is True
        # In Python, False == 0 is True, so this actually matches in strict mode
        assert _match_single_value("name", False, 0, strict=True) is True
        assert _match_single_value("name", True, 1, strict=True) is True  # True == 1 is also True
        assert _match_single_value("name", True, 2, strict=True) is False  # But True != 2

    def test_complex_data_types_lists(self):
        """Test with list data types."""
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        list3 = [3, 2, 1]

        assert _match_single_value("name", list1, list2, strict=True) is True
        assert _match_single_value("name", list1, list3, strict=True) is False
        assert _match_single_value("name", list1, list2, strict=False) is True
        assert _match_single_value("name", list1, list3, strict=False) is False

    def test_complex_data_types_dicts(self):
        """Test with nested dictionary data types."""
        dict1 = {"a": {"b": "c"}}
        dict2 = {"a": {"b": "c"}}
        dict3 = {"a": {"b": "d"}}

        assert _match_single_value("name", dict1, dict2, strict=True) is True
        assert _match_single_value("name", dict1, dict3, strict=True) is False

    def test_strict_mode_type_mismatches(self):
        """Test behavior with different types in strict mode."""
        assert _match_single_value("name", "123", 123, strict=True) is False
        assert _match_single_value("name", 123, "123", strict=True) is False

    def test_case_sensitivity_non_strict(self):
        """Test case sensitivity in non-strict mode for strings."""
        assert _match_single_value("name", "TEST", "test", strict=False) is True
        assert _match_single_value("name", "Test", "EST", strict=False) is True
        assert _match_single_value("name", "lowercase", "CASE", strict=False) is True
        assert _match_single_value("name", "NoMatch", "xyz", strict=False) is False

    def test_whitespace_handling(self):
        """Test whitespace handling in different modes."""
        # Strict mode - exact match required
        assert _match_single_value("name", "test", " test", strict=True) is False
        assert _match_single_value("name", "test ", "test", strict=True) is False

        # Non-strict mode - substring matching
        assert _match_single_value("name", "  test  ", "test", strict=False) is True
        assert _match_single_value("name", "test", "es", strict=False) is True

    def test_unicode_strings(self):
        """Test with unicode strings."""
        assert _match_single_value("name", "café", "café", strict=True) is True
        assert _match_single_value("name", "CAFÉ", "café", strict=False) is True
        assert _match_single_value("name", "测试", "试", strict=False) is True
        assert _match_single_value("name", "🚀 rocket", "rocket", strict=False) is True

    def test_string_representations_of_types(self):
        """Test string item_value with string filter_value containing representations of other types."""
        # This should work fine since both are strings
        assert _match_single_value("name", "123456", "123", strict=False) is True
        assert _match_single_value("name", "test123test", "123", strict=False) is True
        assert _match_single_value("name", "true", "true", strict=False) is True
        assert _match_single_value("name", "false", "alse", strict=False) is True
