# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes storage quantities in the Gym host's settings, checked where Evaluator plans a job.

The Gym host reads the same settings with ``sandboxed_gym.quantity`` and ``sandboxed_gym.host.opensandbox``.
Evaluator can't import those: the ``nemo-helix`` wheel installs ``nemo-sandboxed-gym`` from PyPI at its own
released version, which predates both. ``tests/test_sandbox_quantity.py`` keeps these copies in step with the
Gym host's. Import from ``sandboxed_gym`` instead once the wheel pins a release that has them.
"""

import math
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

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

EPHEMERAL_STORAGE = "ephemeral-storage"


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


def host_disk_request_gib(resources: Mapping[str, str] | None, create: Mapping[str, Any]) -> int | None:
    """The host's ephemeral-storage request in whole GiB, rounded up, or ``None`` when it asks for none.

    The sandbox API takes the request in GiB. An operator's own ``ephemeral-storage`` limit in
    ``create.resource`` must not be below it, or Kubernetes rejects the pod, which surfaces only as a create
    call that never finishes.
    """
    if not resources or resources.get(EPHEMERAL_STORAGE) in (None, ""):
        return None
    request = str(resources[EPHEMERAL_STORAGE])
    try:
        gib = math.ceil(quantity_bytes(request) / 2**30)
    except ValueError as exc:
        raise ValueError(f"sandbox resource {EPHEMERAL_STORAGE}: {exc}") from None
    limit = (create.get("resource") or {}).get(EPHEMERAL_STORAGE)
    if limit is not None and gib * 2**30 > quantity_bytes(str(limit)):
        raise ValueError(
            f"sandbox resource {EPHEMERAL_STORAGE} requests {request} ({gib}Gi, the sandbox API's whole-GiB "
            f"request) but the create options limit it to {limit}"
        )
    return gib
