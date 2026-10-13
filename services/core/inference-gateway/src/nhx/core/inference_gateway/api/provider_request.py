# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

from jinja2 import Environment

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer {{ auth_secret }}"
# Renders HTTP headers, not HTML. Autoescape would corrupt some secret values.
_JINJA_ENV = Environment(autoescape=False)  # noqa: S701  # nosec B701


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Render a provider authentication header from its validated template."""
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    rendered = _JINJA_ENV.from_string(template).render(auth_secret=secret_value)
    header_name, _, header_value = rendered.partition(": ")
    return header_name, header_value
