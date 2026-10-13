# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation logic for entity fields."""

import re
from typing import Optional

from nhx.common.entities.constants import REGEX_WORD_CHARACTER_DOT_DASH
from nhx.customization_common.schemas.file_io import FILESET_PROTOCOL, FileSetRef

_NAME_REGEX = re.compile(REGEX_WORD_CHARACTER_DOT_DASH)
_UNSUPPORTED_PROTOCOLS = ("hf://", "ngc://", "s3://", "gs://")


def _normalize_fileset_ref(uri: str) -> str:
    """Parse and return canonical fileset reference (no ``fileset://`` prefix)."""
    normalized = uri.strip()
    for prefix in _UNSUPPORTED_PROTOCOLS:
        if normalized.startswith(prefix):
            raise ValueError(
                f"Unsupported dataset URI protocol. Use 'workspace/name' or 'name' (resolved in the job workspace). Got: {uri}",
            )
    if normalized.startswith(FILESET_PROTOCOL):
        normalized = normalized[len(FILESET_PROTOCOL) :]
    ref = FileSetRef.model_validate(normalized)
    if not _NAME_REGEX.match(ref.name):
        raise ValueError(
            f"Invalid dataset name: '{ref.name}'. Entity names must contain only word characters, dots, and hyphens.",
        )
    return str(ref)


def validate_fileset_uri(uri: str) -> str:
    """Validate a fileset reference as ``workspace/name`` or ``name``.

    The job path ``workspace`` is used when the reference is a bare name.
    A legacy ``fileset://`` prefix is accepted and stripped.
    """
    return _normalize_fileset_ref(uri)


def validate_optional_fileset_uri(uri: Optional[str]) -> Optional[str]:
    """Validate fileset reference, allowing None."""
    if uri is None:
        return None
    return validate_fileset_uri(uri)
