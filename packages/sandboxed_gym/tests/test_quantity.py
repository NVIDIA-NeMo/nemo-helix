# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sandboxed_gym.quantity import quantity_bytes, quantity_gib_rounded_up


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("30Gi", 30 * 2**30),
        # Kubernetes' G is decimal; GNU truncate's G is binary. The setup gets bytes, so neither is misread.
        ("30G", 30 * 10**9),
        ("512Mi", 512 * 2**20),
        ("1.5Gi", 3 * 2**29),
        ("1073741824", 2**30),
        ("2Ti", 2 * 2**40),
    ],
)
def test_a_quantity_is_its_exact_byte_count(quantity: str, expected: int) -> None:
    assert quantity_bytes(quantity) == expected


@pytest.mark.parametrize("quantity", ["", "0", "0Gi", "-1Gi", "30 Gi", "30gi", "1e9", "30GB", "0.5", "Gi"])
def test_anything_but_a_positive_whole_byte_quantity_is_refused(quantity: str) -> None:
    with pytest.raises(ValueError):
        quantity_bytes(quantity)


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [("40Gi", 40), ("40G", 38), ("1", 1), ("30Gi", 30), ("1.5Gi", 2)],
)
def test_whole_gib_round_up_so_a_request_never_asks_for_less(quantity: str, expected: int) -> None:
    assert quantity_gib_rounded_up(quantity) == expected
