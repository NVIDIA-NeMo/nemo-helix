# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-processor pixel bounds shared by the Automodel and RL backends.

``longest_edge`` is the maximum pixel count and matches ``max_pixels``. ``shortest_edge``
is the minimum pixel count. Both are areas, not side lengths.
"""

from __future__ import annotations

DEFAULT_SHORTEST_EDGE = 65_536
"""Platform floor for the vision processor when a ceiling is set and ``shortest_edge`` is not.

65_536 pixels (256x256) is the ``shortest_edge`` in the Qwen3-VL processor config, which the Super VL
and Qwen3-VL recipes use. A ceiling at or below this floor gets half the ceiling as its floor instead.
"""


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
        # A small ceiling would sit below the platform floor, so halve it instead.
        shortest_edge = DEFAULT_SHORTEST_EDGE if longest_edge > DEFAULT_SHORTEST_EDGE else longest_edge // 2
        if shortest_edge < 1:
            raise ValueError("max_pixels or image_size.longest_edge must be at least 2.")
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
    if max_pixels is not None and max_pixels < 2:
        # image_size_bounds derives the floor as max_pixels // 2, so 1 would yield 0. Fail before the job exists.
        raise ValueError("max_pixels must be at least 2.")


def check_seq_length_fits_image(
    max_seq_length: int,
    max_pixels: int | None,
    longest_edge: int | None,
    vision_patch_size: int | None,
) -> None:
    """Reject a token cap that cannot hold one image at the pixel ceiling.

    This is a lower bound. Rows also carry text tokens (prompt and answer), which the platform cannot
    measure here, so a cap just above the image tokens can still truncate the text. Leave room for it.

    ``vision_patch_size`` is the side length, in pixels, of one image token from the model spec.
    Without it the token count is unknown, so the cap is left to the recipe.
    """
    ceiling = max_pixels if max_pixels is not None else longest_edge
    if ceiling is None or not vision_patch_size:
        return
    image_tokens = ceiling // (vision_patch_size * vision_patch_size)
    if image_tokens and max_seq_length <= image_tokens:
        raise ValueError(
            f"max_seq_length ({max_seq_length}) must be larger than the {image_tokens} tokens "
            f"one image can use at this pixel ceiling (one token per {vision_patch_size}x{vision_patch_size} pixels). "
            "Rows also hold text tokens, so leave room for them."
        )
