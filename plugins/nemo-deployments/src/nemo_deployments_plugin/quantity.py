# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes storage quantities (e.g. ``32Gi``), parsed without the optional kubernetes client."""

from __future__ import annotations

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
    """Bytes in a Kubernetes quantity such as ``32Gi`` or ``8G``; raises ``ValueError`` otherwise."""
    match = _QUANTITY.fullmatch(quantity.strip())
    if match is None:
        raise ValueError(f"{quantity!r} is not a Kubernetes quantity such as '32Gi' or '512Mi'")
    number, suffix = match.groups()
    return int(Decimal(number) * _FACTORS[suffix or ""])
