# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from fastapi import HTTPException

try:
    from scaled_evals.api.tenancy import _ADMIN_PROBE, caller_is_admin
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_caller_is_admin_probes_the_platform_authorizer() -> None:
    probed: list[tuple[str, str]] = []

    def authorizer(status: int | None):
        async def authorize(method: str, path: str) -> None:
            probed.append((method, path))
            if status is not None:
                raise HTTPException(status_code=status)

        return authorize

    assert asyncio.run(caller_is_admin(authorizer(None))) is True
    assert asyncio.run(caller_is_admin(authorizer(403))) is False
    with pytest.raises(HTTPException) as raised:
        asyncio.run(caller_is_admin(authorizer(503)))
    assert raised.value.status_code == 503
    assert probed == [_ADMIN_PROBE] * 3
