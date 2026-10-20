# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared output-naming helpers for the input → canonical spec transform.

Both backends generate the output entity/fileset name the same way when the
submitter omits one: slugified model basename + slugified dataset basename +
a short random suffix. Those helpers live here; the schema-bound assembly of
the canonical output (which fields exist, how the output type is inferred)
stays in each backend's ``transform.py``.
"""

from __future__ import annotations

import re
import uuid

from nemo_helix_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nhx.common.entities.utils import parse_entity_ref

_OUTPUT_NAME_RE = re.compile(NAME_PATTERN)

_MAX_PREFIX_LEN = 50
_HEX_LEN = 12
_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(token: str) -> str:
    """Reduce an arbitrary string to a platform-safe name segment.

    Runs of characters outside ``[A-Za-z0-9_-]`` collapse to a single ``-``;
    leading/trailing ``-`` are trimmed; an empty result falls back to ``"x"``
    so the segment is never empty. E.g. ``"Meta-Llama-3.1 8B"`` -> ``"Meta-Llama-3-1-8B"``.
    """
    cleaned = _NAME_SAFE_RE.sub("-", token).strip("-")
    return cleaned or "x"


def random_suffix(prefix: str) -> str:
    """Append a short random hex suffix to a length-capped prefix."""
    truncated = prefix[:_MAX_PREFIX_LEN].rstrip("-")
    return f"{truncated}-{uuid.uuid4().hex[:_HEX_LEN]}"


def model_basename(model_ref: str, workspace: str) -> str:
    """Slugified entity name of a model ref (handles ``workspace/name`` and ``name``)."""
    return slugify(parse_entity_ref(model_ref, workspace).name)


def dataset_basename(uri: str) -> str:
    """Slugified last segment of a fileset ref.

    Handles an optional protocol prefix (e.g. ``fileset://``) and
    ``workspace/name`` form.
    """
    cleaned = uri.split("://", 1)[-1]
    last = cleaned.rsplit("/", 1)[-1] or cleaned
    return slugify(last)


def generated_output_name(model_ref: str, dataset_ref: str, workspace: str) -> str:
    """Default output entity/fileset name used when the submitter omits one."""
    return random_suffix(f"{model_basename(model_ref, workspace)}-{dataset_basename(dataset_ref)}")


def require_output_fileset_name(name: str) -> str:
    """Reject an output model name that cannot be the fileset created for its weights.

    The fileset is created only after training. A name that fails this check must stop the job
    at submit, before any step is scheduled.
    """
    if _OUTPUT_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"output model name {name!r} is not a valid fileset name, so the job was not created. "
            f"A valid fileset name: {NAME_PATTERN_DESCRIPTION}"
        )
    return name
