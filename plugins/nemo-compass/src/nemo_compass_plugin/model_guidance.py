# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonblocking model guidance for Compass analysis."""

import re


def model_warning(model: str) -> str | None:
    """Recognize the tested 3.5 Lightning 30B model, including Helix slug names."""
    if re.search(
        r"(?:^|/)(?:nvidia/|(?:nvidia[-_]){1,2})nemotron[-_]3[._-]5[-_]lightning[-_]30b", model, re.IGNORECASE
    ):
        return (
            f"{model} has known tool-calling failures in Compass. "
            "Try NVIDIA Nemotron 3.5 Super VL instead. "
            "Analysis will continue with your selected model."
        )
    return None
