# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-processor pixel bounds shared by the Automodel and RL backends.

``longest_edge`` is the maximum pixel count and matches ``max_pixels``. ``shortest_edge``
is the minimum pixel count. Both are areas, not side lengths.
"""

from __future__ import annotations

DEFAULT_SHORTEST_EDGE = 65_536
"""Platform floor for the vision processor when a ceiling is set and ``shortest_edge`` is not."""


def image_size_bounds(
    max_pixels: int | None,
    longest_edge: int | None,
    shortest_edge: int | None,
) -> dict[str, int] | None:
    """Return the processor ``size`` dict, or ``None`` when no bound is set."""
    if max_pixels is not None:
        longest_edge = max_pixels
    if longest_edge is None and shortest_edge is None:
        return None
    if longest_edge is None:
        raise ValueError("max_pixels or image_size.longest_edge is required to set the vision processor size.")
    if shortest_edge is None:
        shortest_edge = DEFAULT_SHORTEST_EDGE
    return {"longest_edge": longest_edge, "shortest_edge": shortest_edge}


def check_image_size_edges(longest_edge: int | None, shortest_edge: int | None) -> None:
    """Both edges are required, and ``shortest_edge`` must be below ``longest_edge``."""
    if longest_edge is None or shortest_edge is None:
        raise ValueError("image_size needs both longest_edge and shortest_edge.")
    if shortest_edge >= longest_edge:
        raise ValueError("image_size.shortest_edge must be less than longest_edge.")


def check_max_pixels_or_image_size(max_pixels: int | None, image_size_set: bool) -> None:
    """``max_pixels`` and ``image_size`` are two ways to set the same ceiling, so only one may be set."""
    if max_pixels is not None and image_size_set:
        raise ValueError("Set max_pixels or image_size, not both.")
