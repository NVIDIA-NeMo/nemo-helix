# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ProviderNameValidator in the prompts module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_helix_ext.ui.prompts import (
    ProviderNameValidator,
    _normalize_choices,
    _resolve_select_response,
    prompt_search_select,
)
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.shortcuts import CompleteStyle
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


def test_select_response_resolves_number_value_and_label() -> None:
    choices = _normalize_choices([("default/model-a", "Model A"), ("default/model-b", "Model B")])

    assert _resolve_select_response("2", choices) == "default/model-b"
    assert _resolve_select_response("default/model-a", choices) == "default/model-a"
    assert _resolve_select_response("model b", choices) == "default/model-b"
    assert _resolve_select_response("missing", choices) is None


def test_search_select_empty_response_requires_default_in_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = ["", "2"]

    class FakePromptSession:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def prompt(self, message, **kwargs) -> str:  # type: ignore[no-untyped-def]
            to_formatted_text(message)
            return responses.pop(0)

    monkeypatch.setattr("nemo_helix_ext.ui.prompts.PromptSession", FakePromptSession)

    result = prompt_search_select(
        "Choose:",
        [("default/model-a", "Model A"), ("default/model-b", "Model B")],
        default="default/missing",
    )

    assert result == "default/model-b"


def test_search_select_escapes_default_label_and_replaces_full_input(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions: list[Any] = []
    prompt_kwargs: list[dict[str, Any]] = []

    class FakePromptSession:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            sessions.append(self)

        def prompt(self, message, **kwargs) -> str:  # type: ignore[no-untyped-def]
            to_formatted_text(message)
            prompt_kwargs.append(kwargs)
            return ""

    monkeypatch.setattr("nemo_helix_ext.ui.prompts.PromptSession", FakePromptSession)

    result = prompt_search_select(
        "Choose:",
        [("default/model-a", "Model <A&>")],
        default="default/model-a",
    )

    assert result == "default/model-a"
    completer = sessions[0].kwargs["completer"]
    completions = list(completer.get_completions(Document("Model A"), MagicMock()))
    assert completions
    assert completions[0].start_position == -len("Model A")
    assert prompt_kwargs[0]["complete_style"] == CompleteStyle.COLUMN
    assert callable(prompt_kwargs[0]["pre_run"])
