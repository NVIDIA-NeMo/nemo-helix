# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the keyword arguments a job passes to a vision-language processor.

Hugging Face processors drop a keyword they do not read, so a wrong name changes nothing and
raises nothing. The model spec records the keys the processor's config file sets, and a job's
arguments are checked against them.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

LOADER_ARGUMENTS = frozenset({"use_fast"})
"""Arguments of ``AutoProcessor.from_pretrained`` itself, which never appear in a processor config file."""


def check_processor_arguments(arguments: Mapping[str, Any], accepted_keys: Collection[str] | None) -> None:
    """Reject an argument the model's processor does not read. ``None`` means the keys are unknown, so all pass."""
    if accepted_keys is None:
        return
    unknown = sorted(set(arguments) - set(accepted_keys) - LOADER_ARGUMENTS)
    if unknown:
        raise ValueError(
            f"processor_arguments {unknown} are not read by this model's processor and would be ignored. "
            f"The processor config sets: {sorted(accepted_keys)}."
        )
