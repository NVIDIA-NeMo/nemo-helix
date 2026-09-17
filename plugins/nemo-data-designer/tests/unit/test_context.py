# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from data_designer_nemo.context import DataDesignerValidationContext
from data_designer_nemo.errors import NDDInvalidConfigError
from nemo_platform import AsyncNeMoPlatform

LOCAL_PROVIDER_A = "local-provider-a"
LOCAL_PROVIDER_B = "local-provider-b"


@pytest.fixture
def local_providers() -> dict[str, dd.ModelProvider]:
    return {
        LOCAL_PROVIDER_A: dd.ModelProvider(
            name=LOCAL_PROVIDER_A,
            endpoint="http://example.com",
        ),
        LOCAL_PROVIDER_B: dd.ModelProvider(
            name=LOCAL_PROVIDER_B,
            endpoint="http://example.com",
        ),
    }


@contextmanager
def patch_local_lookup(providers: dict[str, dd.ModelProvider] | None = None):
    with patch("data_designer_nemo.model_provider.get_default_providers") as default_lookup:
        if providers:
            default_lookup.return_value = list(providers.values())
        yield default_lookup


def _simple_config(
    *,
    tool_configs: list[dd.ToolConfig] | None = None,
    seed_source: dd.DataFrameSeedSource | None = None,
) -> dd.DataDesignerConfig:
    return dd.DataDesignerConfig(
        columns=[dd.ExpressionColumnConfig(name="value", expr="'ok'")],
        model_configs=[],
        tool_configs=tool_configs or [],
        seed_config=dd.SeedConfig(source=seed_source) if seed_source is not None else None,
    )


async def test_remote_validate_runs_remote_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    config = _simple_config()
    sdk = AsyncMock(spec=AsyncNeMoPlatform)

    def validate_tools(validated_config: dd.DataDesignerConfig) -> None:
        assert validated_config is config
        calls.append("tools")

    async def validate_seed(
        validated_config: dd.DataDesignerConfig, workspace: str, async_sdk: AsyncNeMoPlatform
    ) -> None:
        assert validated_config is config
        assert workspace == u.WORKSPACE_NAME
        assert async_sdk is sdk
        calls.append("seed")

    async def validate_personas(validated_config: dd.DataDesignerConfig, async_sdk: AsyncNeMoPlatform) -> None:
        assert validated_config is config
        assert async_sdk is sdk
        calls.append("personas")

    monkeypatch.setattr("data_designer_nemo.context.validate_no_tool_configs", validate_tools)
    monkeypatch.setattr("data_designer_nemo.context.validate_seed", validate_seed)
    monkeypatch.setattr("data_designer_nemo.context.ensure_nemotron_personas_filesets", validate_personas)

    errors = await DataDesignerValidationContext(sdk, u.WORKSPACE_NAME).validate(config)

    assert errors == []
    assert calls == ["tools", "seed", "personas"]


async def test_remote_validate_rejects_unsupported_seed_config() -> None:
    config = _simple_config(seed_source=dd.DataFrameSeedSource(df=pd.DataFrame(data={"a": [1, 2, 3]})))
    dd_ctx = DataDesignerValidationContext(AsyncMock(spec=AsyncNeMoPlatform), u.WORKSPACE_NAME)

    errors = await dd_ctx.validate(config)

    assert len(errors) == 1
    assert isinstance(errors[0], NDDInvalidConfigError)
    assert "seed data" in str(errors[0])


async def test_remote_validate_aggregates_multiple_failures() -> None:
    """Tool configs *and* an unsupported seed type both surface from a single pass."""
    config = _simple_config(
        tool_configs=[dd.ToolConfig(tool_alias="hello", providers=["provider"])],
        seed_source=dd.DataFrameSeedSource(df=pd.DataFrame(data={"a": [1, 2, 3]})),
    )
    sdk = AsyncMock(spec=AsyncNeMoPlatform)
    dd_ctx = DataDesignerValidationContext(sdk, u.WORKSPACE_NAME)

    errors = await dd_ctx.validate(config)

    messages = [str(e) for e in errors]
    assert all(isinstance(e, NDDInvalidConfigError) for e in errors)
    # We expect at least the tool-config and seed-type messages; both must surface
    # in a single pass (no short-circuiting on the first failure).
    assert any("Tool configs" in m for m in messages)
    assert any("seed data" in m or "df" in m for m in messages)
    assert len(errors) >= 2
