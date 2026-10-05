# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_helix_sdk_tools.sdk.vendor.dependency_utils import merge_dependencies, merge_version_specifiers


class TestSimplifySpecifiers:
    def test_keeps_highest_lower_bound(self):
        result = merge_version_specifiers(">=1.0.0", ">=2.0.0")
        assert result == ">=2.0.0"

    def test_keeps_lowest_upper_bound(self):
        result = merge_version_specifiers("<3.0.0", "<2.0.0")
        assert result == "<2.0.0"

    def test_combines_lower_and_upper_bounds(self):
        result = merge_version_specifiers(">=1.0.0,<3.0.0", ">=2.0.0")
        assert result == ">=2.0.0,<3.0.0"

    def test_multiple_lower_bounds_keeps_highest(self):
        result = merge_version_specifiers(">=1.0.0,>=1.5.0", ">=2.0.0")
        assert result == ">=2.0.0"

    def test_multiple_upper_bounds_keeps_lowest(self):
        result = merge_version_specifiers("<5.0.0,<4.0.0", "<3.0.0")
        assert result == "<3.0.0"

    def test_prefers_gt_over_ge_for_same_version(self):
        # > is more restrictive than >=, so > should win
        result = merge_version_specifiers(">1.0.0", ">=1.0.0")
        assert result == ">1.0.0"

    def test_prefers_lt_over_le_for_same_version(self):
        # < is more restrictive, so when versions are equal, < should win
        result = merge_version_specifiers("<=2.0.0", "<2.0.0")
        assert result == "<2.0.0"

    def test_exact_version(self):
        result = merge_version_specifiers("==1.5.0", ">=1.0.0")
        assert "==1.5.0" in result
        assert ">=1.0.0" in result

    def test_preserves_other_specifiers(self):
        result = merge_version_specifiers(">=1.0.0,!=1.5.0", ">=2.0.0")
        assert ">=2.0.0" in result
        assert "!=1.5.0" in result

    def test_empty_specifiers(self):
        result = merge_version_specifiers("", ">=1.0.0")
        assert result == ">=1.0.0"

    def test_both_empty(self):
        result = merge_version_specifiers("", "")
        assert result == ""


class TestMergeDependencies:
    def test_no_overlap_combines_all(self):
        existing = ["requests>=2.0.0", "numpy>=1.0.0"]
        new = ["pandas>=1.0.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 3
        assert "requests>=2.0.0" in result
        assert "numpy>=1.0.0" in result
        assert "pandas>=1.0.0" in result

    def test_same_package_merges_specifiers(self):
        existing = ["requests>=2.0.0,<3.0.0"]
        new = ["requests>=2.5.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 1
        assert "requests" in result[0]
        assert ">=2.5.0" in result[0]
        assert "<3.0.0" in result[0]

    def test_preserves_original_format_when_no_conflict(self):
        existing = ["requests>=2.0.0, <3.0.0"]  # Note: space after comma
        new = ["numpy>=1.0.0"]
        result = merge_dependencies(existing, new)
        # Original string should be preserved
        assert "requests>=2.0.0, <3.0.0" in result

    def test_case_insensitive_package_names(self):
        existing = ["Requests>=2.0.0"]
        new = ["requests>=2.5.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 1

    def test_merges_extras(self):
        existing = ["package[extra1]>=1.0.0"]
        new = ["package[extra2]>=1.0.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 1
        assert "extra1" in result[0]
        assert "extra2" in result[0]

    def test_identical_deps_keeps_first(self):
        existing = ["requests>=2.0.0"]
        new = ["requests>=2.0.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 1
        assert result[0] == "requests>=2.0.0"

    def test_empty_existing(self):
        existing: list[str] = []
        new = ["requests>=2.0.0"]
        result = merge_dependencies(existing, new)
        assert result == ["requests>=2.0.0"]

    def test_empty_new(self):
        existing = ["requests>=2.0.0"]
        new: list[str] = []
        result = merge_dependencies(existing, new)
        assert result == ["requests>=2.0.0"]

    def test_both_empty(self):
        result = merge_dependencies([], [])
        assert result == []

    def test_real_world_example(self):
        # Simulating SDK deps vs vendored package deps
        existing = [
            "httpx>=0.23.0, <1",
            "pydantic>=1.9.0, <3",
            "typing-extensions>=4.10, <5",
            "anyio>=3.5.0, <5",
            "distro>=1.7.0, <2",
            "sniffio",
        ]
        new = [
            "httpx>=0.23.0",
            "pydantic>=1.9.0",
            "typing-extensions>=4.10",
            "anyio>=4.0.0",  # Higher lower bound
            "distro>=1.7.0",
            "sniffio",
        ]
        result = merge_dependencies(existing, new)

        # Find anyio in results
        anyio_dep = next(d for d in result if "anyio" in d.lower())
        # Should have the higher lower bound (>=4.0.0) and keep the upper bound (<5)
        assert ">=4.0.0" in anyio_dep or ">=4" in anyio_dep
        assert "<5" in anyio_dep

    def test_complex_specifiers_with_tilde(self):
        existing = ["package~=1.4.2"]
        new = ["package>=1.5.0"]
        result = merge_dependencies(existing, new)
        assert len(result) == 1
        # Both specifiers should be present
        assert "~=1.4.2" in result[0]
        assert ">=1.5.0" in result[0]
