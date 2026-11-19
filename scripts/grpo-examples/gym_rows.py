# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Gym rollout-row helpers for the GRPO example scripts.

``write_rows`` is the row numbering ``prepare_math_with_judge`` used to do inline.
``image_message`` is the Responses API part list an image prompt needs: an
``input_image`` part, then an ``input_text`` part. Image bytes travel as a
``data:`` URI because a Gym row does not resolve a fileset-relative path.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any


def pil_to_data_uri(image: Any, fmt: str = "PNG") -> str:
    """Encode a PIL image as a ``data:image/...;base64,...`` URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def image_message(text: str, images: list[str]) -> dict[str, Any]:
    """One user turn: each image URI, then the text."""
    content: list[dict[str, Any]] = [{"type": "input_image", "image_url": uri, "detail": "auto"} for uri in images]
    content.append({"type": "input_text", "text": text})
    return {"role": "user", "content": content}


def write_rows(path: Path, rows: list[dict[str, Any]], agent_ref: dict[str, str], repeats: int = 1) -> int:
    """Write rows with ``agent_ref`` and a stable ``task_idx``, repeating each ``repeats`` times.

    ``task_idx`` is not read by the resources server. Gym sorts rollouts by it, and the
    platform's dataset validators expect an int, so number the emitted rows rather than
    the source rows.
    """
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for _ in range(repeats):
                handle.write(json.dumps({**row, "task_idx": written, "agent_ref": agent_ref}) + "\n")
                written += 1
    return written
