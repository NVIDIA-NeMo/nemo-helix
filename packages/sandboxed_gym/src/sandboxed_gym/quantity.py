# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes storage quantities, as the Gym host's operator settings write them."""

import math
import re
from decimal import Decimal

_FACTORS = {
    "": 1,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "Pi": 2**50,
    "Ei": 2**60,
    "k": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "E": 10**18,
}
_QUANTITY = re.compile(r"([0-9]+(?:\.[0-9]+)?)(Ki|Mi|Gi|Ti|Pi|Ei|k|M|G|T|P|E)?")


def quantity_bytes(quantity: str) -> int:
    """Bytes in a Kubernetes quantity such as ``30Gi`` or ``32G``; raises ``ValueError`` otherwise.

    Suffixes mean what they mean to Kubernetes: ``G`` is 10**9 bytes and ``Gi`` is 2**30. The result must be a
    positive whole number of bytes. Exponent forms such as ``1e9`` are not accepted.
    """
    match = _QUANTITY.fullmatch(quantity)
    if match is None:
        raise ValueError(f"{quantity!r} is not a Kubernetes quantity such as '30Gi' or '512Mi'")
    number, suffix = match.groups()
    size = Decimal(number) * _FACTORS[suffix or ""]
    if size <= 0 or size != size.to_integral_value():
        raise ValueError(f"{quantity!r} must be a positive whole number of bytes")
    return int(size)


def quantity_gib_rounded_up(quantity: str) -> int:
    """Whole GiB covering a Kubernetes quantity, for requests the sandbox API takes in GiB."""
    return math.ceil(quantity_bytes(quantity) / 2**30)
