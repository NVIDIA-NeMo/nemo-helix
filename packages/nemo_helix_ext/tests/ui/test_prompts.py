# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ProviderNameValidator in the prompts module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nemo_helix_ext.ui.prompts import ProviderNameValidator
from prompt_toolkit.validation import ValidationError


def _make_document(text: str) -> MagicMock:
    doc = MagicMock()
    doc.text = text
    return doc


@pytest.fixture
def validator() -> ProviderNameValidator:
    return ProviderNameValidator()


class TestProviderNameValidator:
    @pytest.mark.parametrize(
        "name",
        [
            "my-provider",
            "my.provider_1",
            "ab",
            "a1",
        ],
    )
    def test_accepts_valid_names(self, validator: ProviderNameValidator, name: str) -> None:
        validator.validate(_make_document(name))

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "   ",
            "my provider",
            "my@provider!",
            "my/provider",
            "23skidoo",
            "MyProvider",
            "a",
            "a--b",
            "ab-",
            " my-provider ",
        ],
    )
    def test_rejects_invalid_names(self, validator: ProviderNameValidator, name: str) -> None:
        with pytest.raises(ValidationError):
            validator.validate(_make_document(name))

    def test_error_message_describes_name_rules(self, validator: ProviderNameValidator) -> None:
        with pytest.raises(ValidationError, match=r"lowercase letter"):
            validator.validate(_make_document("bad name!"))
