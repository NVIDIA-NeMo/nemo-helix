# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``AnonymizerService.get_exception_handlers``."""

from __future__ import annotations

import pytest
from nemo_anonymizer_plugin.service import AnonymizerService
from starlette import status


def test_type_error_maps_to_422() -> None:
    """ASTD-328: a bare TypeError (e.g. the upstream replace-kind Discriminator) must 422, not 500."""
    handlers = AnonymizerService().get_exception_handlers()

    assert TypeError in handlers


@pytest.mark.asyncio
async def test_type_error_handler_returns_422_response() -> None:
    handlers = AnonymizerService().get_exception_handlers()
    handler = handlers[TypeError]

    response = await handler(None, TypeError("dict is missing a valid 'kind' key: {}"))

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
