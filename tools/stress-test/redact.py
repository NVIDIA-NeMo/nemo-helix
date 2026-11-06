# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove credential material from text that can appear in a report."""

from __future__ import annotations

import json
import re

_SECRET_KEY = re.compile(r'("(?:token|secret|password|authorization|access_key)"\s*:\s*")([^"]*)(")')
_SECRET_WORD = re.compile(r"secret-[A-Za-z0-9_-]+")


def redact(text: str) -> str:
    """Replace credential-shaped values. Timing and status numbers stay."""

    def _hide(match: re.Match[str]) -> str:
        return f"{match.group(1)}[redacted]{match.group(3)}"

    hidden = _SECRET_KEY.sub(_hide, text)
    return _SECRET_WORD.sub("[redacted]", hidden)


def json_line(payload: dict[str, object]) -> str:
    """Serialize one history record without credential keys."""
    blocked = {"token", "secret", "password", "authorization", "access_key", "value"}
    safe = {key: value for key, value in payload.items() if key not in blocked}
    return json.dumps(safe, sort_keys=True)
