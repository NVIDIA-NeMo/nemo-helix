# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_compass_plugin.contracts.insights import (
    InsightsFileError,
    load_insights_document,
    validate_insights_file,
)


def test_validate_insights_file_allows_none_missing_and_mapping_without_records(tmp_path: Path) -> None:
    validate_insights_file(None)
    validate_insights_file(tmp_path / "missing.yaml")
    path = tmp_path / "insights.yaml"
    path.write_text("metadata: retained\n", encoding="utf-8")
    validate_insights_file(path)


def test_load_insights_document_returns_validated_mapping(tmp_path: Path) -> None:
    path = tmp_path / "insights.yaml"
    path.write_text("insights:\n  - id: one\n", encoding="utf-8")

    assert load_insights_document(path) == {"insights": [{"id": "one"}]}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- id: one\n", "YAML mapping"),
        ("insights: null\n", "`insights` must be a list"),
        ("insights: 42\n", "`insights` must be a list"),
        ("insights:\n  - id: one\n  - broken\n", "item 2 must be a YAML mapping"),
    ],
)
def test_invalid_insights_shapes_are_actionable(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "insights.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(InsightsFileError, match=message):
        load_insights_document(path)


def test_invalid_yaml_error_includes_path_on_one_line(tmp_path: Path) -> None:
    path = tmp_path / "insights.yaml"
    path.write_text("insights: [\n", encoding="utf-8")

    with pytest.raises(InsightsFileError, match="valid YAML") as exc_info:
        load_insights_document(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "\n" not in message


def test_invalid_utf8_is_actionable_without_raw_chain(tmp_path: Path) -> None:
    path = tmp_path / "insights.yaml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(InsightsFileError, match="UTF-8") as exc_info:
        load_insights_document(path)

    assert exc_info.value.__cause__ is None


def test_generic_os_error_uses_neutral_could_not_be_read_wording(tmp_path: Path) -> None:
    path = tmp_path / "insights.yaml"
    path.mkdir()

    with pytest.raises(InsightsFileError, match="could not be read") as exc_info:
        load_insights_document(path)

    message = str(exc_info.value)
    assert "UTF-8" not in message
    assert exc_info.value.__cause__ is None


def test_validate_treats_disappearance_at_read_boundary_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "insights.yaml"
    path.write_text("insights: []\n", encoding="utf-8")

    def _vanished(self: Path, *args: object, **kwargs: object) -> str:
        raise FileNotFoundError(2, "No such file or directory", str(self))

    monkeypatch.setattr(Path, "read_text", _vanished)

    validate_insights_file(path)
