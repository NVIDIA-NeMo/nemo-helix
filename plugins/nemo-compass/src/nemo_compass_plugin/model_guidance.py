# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonblocking model guidance for Compass analysis."""

import re


def model_warning(model: str) -> str | None:
    """Recognize Lightning in a model reference or resolved served-model name."""
    if re.search(r"\bnemotron\b.*\blightning\b", model, re.IGNORECASE):
        return (
            f"{model} has known tool-calling failures in Compass. "
            "Try NVIDIA Nemotron 3.5 Super VL instead. "
            "Analysis will continue with your selected model."
        )
    return None
