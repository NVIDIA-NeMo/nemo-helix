# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-entity and adapter string validation for the models service.

Model reference regex and parsing live here so Pydantic validators can use Python ``re`` lookarounds
and so adapter model-ref rules stay with the models API, not in ``nhx_common``.
"""

import re

from nhx.common.entities.client import parse_qualified_name

# Per-segment rules match ``nhx.common.entities.constants.NAME_PATTERN`` (one segment of workspace/model_name).
_NAME_SEGMENT = r"[a-z](?!.*--)[a-z0-9\-@.+_]{1,62}(?<!-)"

MODEL_REF_MAX_LEN = 127  # 63 + '/' + 63
MODEL_REF_PATTERN = rf"^{_NAME_SEGMENT}(/{_NAME_SEGMENT})?$"
MODEL_REF_PATTERN_DESCRIPTION = (
    "A single name (2-63 characters) or 'workspace/model_name' where each segment is a valid name "
    "(lowercase, digits, hyphens, and temporarily @ . + _; no leading/trailing or consecutive hyphens). "
    "If one slash, both sides must be non-empty."
)
_MODEL_REF_RE = re.compile(MODEL_REF_PATTERN)


def is_valid_model_ref(value: str) -> bool:
    """True if *value* matches :data:`MODEL_REF_PATTERN` (entity NAME rules per segment)."""
    return _MODEL_REF_RE.fullmatch(value) is not None


def parse_model_ref(name: str, adapter_workspace: str) -> tuple[str, str]:
    """Parse a model reference to (model_workspace, model_name) for adapter APIs.

    Same as :func:`~nhx.common.entities.client.parse_qualified_name` for unqualified *name*, but
    for qualified values the string must contain exactly one ``/`` (``a/b`` is valid; ``a/b/c`` is not).
    """
    if not name:
        raise ValueError("Model reference is empty")
    if "/" in name and name.count("/") != 1:
        raise ValueError("Invalid model reference: expected 'workspace/model-name' or a single model name")
    model_workspace, model_name = parse_qualified_name(name, default_workspace=adapter_workspace)
    if "/" in name and (not model_workspace or not model_name):
        raise ValueError("Invalid model reference: expected 'workspace/model-name' or a single model name")
    return model_workspace, model_name
