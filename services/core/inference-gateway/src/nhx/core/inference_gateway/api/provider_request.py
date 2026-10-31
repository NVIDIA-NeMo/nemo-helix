# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

import re

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer <auth_secret>"

# ``auth_header_format`` is literal text around exactly one ``<auth_secret>`` placeholder (or the legacy
# ``{{ auth_secret }}`` spelling). It is
# never evaluated by a template engine: providers stored before this was enforced may carry hostile
# templates, so the format is re-validated here on every use. Keep in sync with the models service
# validator (``nhx.core.models.schemas._validate_auth_header_format``).
_AUTH_SECRET_PLACEHOLDER_RE = re.compile(r"<auth_secret>|\{\{[ \t]*auth_secret[ \t]*\}\}")
_AUTH_HEADER_FORBIDDEN_CHARS = frozenset("{}\r\n")


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Build a provider authentication header from its ``auth_header_format``.

    Raises:
        ValueError: If the format is not literal text with exactly one ``<auth_secret>``
            (or legacy ``{{ auth_secret }}``) placeholder. The offending format is not echoed back.
    """
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    # Splitting on the placeholder yields [prefix, suffix] iff it appears exactly once.
    parts = _AUTH_SECRET_PLACEHOLDER_RE.split(template)
    if len(parts) != 2 or any(_AUTH_HEADER_FORBIDDEN_CHARS.intersection(part) for part in parts):
        raise ValueError(
            "Provider auth_header_format is invalid: it must contain exactly one '<auth_secret>' "
            "(or legacy '{{ auth_secret }}') placeholder"
        )
    rendered = parts[0] + secret_value + parts[1]
    header_name, _, header_value = rendered.partition(": ")
    return header_name, header_value
