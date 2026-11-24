# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.customization_common.image_size import image_size_bounds


@pytest.mark.parametrize(
    ("max_pixels", "shortest_edge"),
    [(32_768, 16_384), (65_536, 32_768)],
)
def test_ceiling_at_or_below_the_default_floor_gets_half_as_its_floor(max_pixels: int, shortest_edge: int) -> None:
    assert image_size_bounds(max_pixels, None, None) == {"longest_edge": max_pixels, "shortest_edge": shortest_edge}


def test_ceiling_too_small_for_a_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        image_size_bounds(1, None, None)
