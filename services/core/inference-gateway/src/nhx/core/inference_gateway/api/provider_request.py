# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

import re

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer {{ auth_secret }}"

# ``auth_header_format`` is literal text around exactly one ``{{ auth_secret }}`` placeholder. It is
# never evaluated by a template engine: providers stored before this was enforced may carry hostile
# templates, so the format is re-validated here on every use. Keep in sync with the models service
# validator (``nhx.core.models.schemas._AUTH_HEADER_FORMAT_RE``).
_AUTH_HEADER_FORMAT_RE = re.compile(r"([^{}\r\n]*)\{\{\s*auth_secret\s*\}\}([^{}\r\n]*)")


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Build a provider authentication header from its ``auth_header_format``.

    Raises:
        ValueError: If the format is not literal text with exactly one ``{{ auth_secret }}``
            placeholder. The offending format is not echoed back.
    """
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    match = _AUTH_HEADER_FORMAT_RE.fullmatch(template)
    if match is None:
        raise ValueError(
            "Provider auth_header_format is invalid: it must contain exactly one '{{ auth_secret }}' placeholder"
        )
    rendered = match.group(1) + secret_value + match.group(2)
    header_name, _, header_value = rendered.partition(": ")
    return header_name, header_value
