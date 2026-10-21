# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural contract for the local Insights YAML document."""

from pathlib import Path
from typing import Any

import yaml


class InsightsFileError(ValueError):
    """A shared Insights file is unreadable or structurally invalid."""


def _read_and_validate(path: Path) -> dict[str, Any]:
    """Read, parse, and shape-check one Insights YAML document.

    The read is the single existence boundary: a missing file surfaces as a
    bare ``FileNotFoundError`` (not wrapped) so callers can each decide
    whether that counts as absent input or a hard error, without a separate
    ``stat()`` racing the actual read.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeError as exc:
        raise InsightsFileError(f"insights file {path} is not valid UTF-8: {exc}") from None
    except OSError as exc:
        raise InsightsFileError(f"insights file {path} could not be read: {exc}") from None
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        detail = " ".join(str(exc).split())
        raise InsightsFileError(f"insights file {path} must contain valid YAML: {detail}") from None
    if not isinstance(payload, dict):
        raise InsightsFileError(f"insights file {path} must contain a YAML mapping at its root")
    if "insights" in payload:
        records = payload["insights"]
        if not isinstance(records, list):
            raise InsightsFileError(f"insights file {path}: `insights` must be a list")
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise InsightsFileError(f"insights file {path}: `insights` item {index} must be a YAML mapping")
    return dict(payload)


def load_insights_document(path: Path) -> dict[str, Any]:
    """Read and validate one existing UTF-8 Insights YAML document."""
    try:
        return _read_and_validate(path)
    except FileNotFoundError as exc:
        raise InsightsFileError(f"insights file {path} could not be read: {exc}") from None


def validate_insights_file(path: Path | None) -> None:
    """Validate an existing file; allow absent optional output files."""
    if path is None:
        return
    try:
        _read_and_validate(path)
    except FileNotFoundError:
        return
