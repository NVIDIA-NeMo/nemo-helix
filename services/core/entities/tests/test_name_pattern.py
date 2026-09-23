# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RFC 1035-like NAME_PATTERN validation.

TODO(#3530): Remove @, ., +, _ once versioning is implemented
Note: The pattern temporarily allows @, ., +, and _ characters to support
predefined targets like 'llama-3.2-3b-instruct@v1.0.0+A100'. These will be
removed once versioning is properly implemented.
"""

import re

import pytest
from nhx.common.entities.constants import NAME_PATTERN


class TestNamePattern:
    """Test the NAME_PATTERN regex used for validating entity, project, and workspace names."""

    @pytest.fixture
    def pattern(self):
        """Compile the NAME_PATTERN regex for testing."""
        return re.compile(NAME_PATTERN)

    # Valid names
    @pytest.mark.parametrize(
        "name",
        [
            "ab",  # minimum length (2 chars)
            "test",
            "test-config",
            "test-config-123",
            "a1",
            "my-app",
            "api-v2",
            "model-123",
            "a" * 63,  # maximum length (63 chars)
            "test-1-2-3",
            "z9",
            "config-a-b-c-123",
            # TODO(#3530): Remove @, ., +, _ once versioning is implemented
            "test_config",  # underscore
            "test.config",  # period
            "test@config",  # at sign
            "test+config",  # plus
            "llama-3.2-3b-instruct@v1.0.0+a100",  # realistic predefined target
            "model_v1.0",  # combination
            "api@v2.1+beta_3",  # multiple special chars
        ],
    )
    def test_valid_names(self, pattern, name):
        """Test that valid names match the pattern."""
        assert pattern.match(name), f"Expected '{name}' to be valid"

    # Invalid: Too short
    @pytest.mark.parametrize(
        "name",
        [
            "",  # empty
            "a",  # 1 character (minimum is 2)
        ],
    )
    def test_invalid_too_short(self, pattern, name):
        """Test that names shorter than 2 characters are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (too short)"

    # Invalid: Too long
    def test_invalid_too_long(self, pattern):
        """Test that names longer than 63 characters are rejected."""
        name = "a" * 64  # 64 characters
        assert not pattern.match(name), f"Expected '{name}' to be invalid (too long)"

    # Invalid: Starts with digit
    @pytest.mark.parametrize(
        "name",
        [
            "1test",
            "123abc",
            "9config",
            "0-test",
        ],
    )
    def test_invalid_starts_with_digit(self, pattern, name):
        """Test that names starting with digits are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (starts with digit)"

    # Invalid: Starts with hyphen or special character
    @pytest.mark.parametrize(
        "name",
        [
            "-test",
            "-config",
            "--test",
            # TODO(#3530): Remove @, ., +, _ once versioning is implemented
            "_test",  # starts with underscore
            ".test",  # starts with period
            "@test",  # starts with at sign
            "+test",  # starts with plus
        ],
    )
    def test_invalid_starts_with_hyphen_or_special(self, pattern, name):
        """Test that names starting with hyphens or special chars are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (starts with hyphen/special)"

    # Invalid: Ends with hyphen
    @pytest.mark.parametrize(
        "name",
        [
            "test-",
            "config-",
            "test--",
        ],
    )
    def test_invalid_ends_with_hyphen(self, pattern, name):
        """Test that names ending with hyphens are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (ends with hyphen)"

    # Invalid: Consecutive hyphens
    @pytest.mark.parametrize(
        "name",
        [
            "test--config",
            "a--b",
            "config---test",
            "test--1",
            "my--app--config",
        ],
    )
    def test_invalid_consecutive_hyphens(self, pattern, name):
        """Test that names with consecutive hyphens are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (consecutive hyphens)"

    # Invalid: Uppercase letters
    @pytest.mark.parametrize(
        "name",
        [
            "Test",
            "TEST",
            "testConfig",
            "test-Config",
        ],
    )
    def test_invalid_uppercase(self, pattern, name):
        """Test that names with uppercase letters are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (contains uppercase)"

    # Invalid: Special characters
    # TODO(#3530): Remove @, ., +, _ once versioning is implemented
    @pytest.mark.parametrize(
        "name",
        [
            "test config",  # space
            "test!config",  # exclamation
            "test#config",  # hash
            "test$config",  # dollar
            "test%config",  # percent
            "test^config",  # caret
            "test&config",  # ampersand
            "test*config",  # asterisk
            "test(config",  # parenthesis
            "test)config",
            "test=config",  # equals
            "test[config",  # bracket
            "test]config",
            "test{config",  # brace
            "test}config",
            "test|config",  # pipe
            "test\\config",  # backslash
            "test/config",  # slash
            "test:config",  # colon
            "test;config",  # semicolon
            "test'config",  # quote
            'test"config',  # double quote
            "test<config",  # less than
            "test>config",  # greater than
            "test?config",  # question mark
            "test,config",  # comma
        ],
    )
    def test_invalid_special_characters(self, pattern, name):
        """Test that names with disallowed special characters are rejected."""
        assert not pattern.match(name), f"Expected '{name}' to be invalid (contains special char)"

    # Edge cases
    def test_edge_case_all_digits_except_start(self, pattern):
        """Test that names can contain all digits except at the start."""
        assert pattern.match("a0000"), "Expected 'a0000' to be valid"

    def test_edge_case_all_hyphens_except_consecutive(self, pattern):
        """Test that names can have many hyphens as long as they're not consecutive."""
        assert pattern.match("a-b-c-d-e"), "Expected 'a-b-c-d-e' to be valid"

    def test_edge_case_exact_min_length(self, pattern):
        """Test that 2-character names are valid."""
        assert pattern.match("ab"), "Expected 'ab' to be valid (exact min length)"

    def test_edge_case_exact_max_length(self, pattern):
        """Test that 63-character names are valid."""
        name = "a" * 63
        assert pattern.match(name), f"Expected '{name}' to be valid (exact max length)"
