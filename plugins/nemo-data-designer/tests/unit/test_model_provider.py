# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_data_designer_plugin.testing.utils as u
import pytest
from data_designer_nemo.model_provider import (
    make_model_provider_registry,
    make_null_registry,
)


@pytest.mark.asyncio
async def test_no_model_configs_returns_none() -> None:
    """``make_model_provider_registry`` returns None for an empty model-config list,
    which the engine treats as 'no LLMs in this config'.
    """
    with u.make_mock_client_context() as client_context:
        registry = await make_model_provider_registry(
            [], client=client_context.async_client, default_workspace=u.WORKSPACE_NAME
        )
    assert registry is None


def test_null_registry() -> None:
    """Configs with no LLM columns get a one-provider 'no-op' registry so the
    Data Designer engine can run without rejecting an empty registry.
    """
    registry = make_null_registry()

    assert len(registry.providers) == 1
