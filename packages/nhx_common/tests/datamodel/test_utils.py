# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

from nhx.common.entities.utils import _remove_optional_from_type


class TestRemoveOptionalFromType:
    def test_remove_optional_with_single_type(self):
        """Test removing Optional wrapper from Optional[str]."""
        optional_str = Optional[str]
        result = _remove_optional_from_type(optional_str)
        assert result is str

    def test_remove_optional_with_int(self):
        """Test removing Optional wrapper from Optional[int]."""
        optional_int = Optional[int]
        result = _remove_optional_from_type(optional_int)
        assert result is int

    def test_remove_optional_with_list_type(self):
        """Test removing Optional wrapper from Optional[List[str]]."""
        optional_list = Optional[List[str]]
        result = _remove_optional_from_type(optional_list)
        assert result is List[str]

    def test_union_with_none_and_multiple_types(self):
        """Test Union with None and multiple other types returns first non-None type."""
        union_type = Union[str, int, None]
        result = _remove_optional_from_type(union_type)
        assert result in (str, int)

    def test_union_without_none(self):
        """Test Union without None type returns original type unchanged."""
        union_type = Union[str, int]
        result = _remove_optional_from_type(union_type)
        # Equality, not identity: CPython 3.14 stopped interning `Union[...]`, so two
        # spellings of the same union are equal but no longer the same object.
        assert result == Union[str, int]

    def test_non_optional_type_unchanged(self):
        """Test that non-Optional types are returned unchanged."""
        result_str = _remove_optional_from_type(str)
        assert result_str is str

        result_int = _remove_optional_from_type(int)
        assert result_int is int

        result_list = _remove_optional_from_type(List[str])
        assert result_list is List[str]

    def test_complex_nested_type(self):
        """Test with complex nested type like Optional[List[Union[str, int]]]."""
        complex_optional = Optional[List[Union[str, int]]]
        result = _remove_optional_from_type(complex_optional)
        assert result is List[Union[str, int]]

    def test_explicit_union_with_none_type(self):
        """Test explicit Union[SomeType, None] format."""
        explicit_optional = Union[str, type(None)]
        result = _remove_optional_from_type(explicit_optional)
        assert result is str

    def test_union_with_none_at_different_positions(self):
        """Test Union where None is not the last argument."""
        union_type = Union[None, str, int]
        result = _remove_optional_from_type(union_type)
        assert result in (str, int)

    def test_remove_optional_with_pep604_union(self):
        """Test removing None from PEP 604 (X | None) style unions."""
        pep604_optional = list[str] | None
        result = _remove_optional_from_type(pep604_optional)
        assert result == list[str]

    def test_pep604_union_without_none(self):
        """Test PEP 604 union without None returns original type."""
        pep604_union = str | int
        result = _remove_optional_from_type(pep604_union)
        assert result == (str | int)
