# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import data_designer.config as dd
import pytest
from data_designer_nemo.errors import NDDError, NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.runnable import resolve_runnable_config


def _config_with_aliased_llm_column() -> dd.DataDesignerConfig:
    """A minimal config that references one ModelConfig via an LLM column."""
    return dd.DataDesignerConfig(
        columns=[
            dd.SamplerColumnConfig(
                name="topic",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=["a", "b"]),
            ),
            dd.LLMTextColumnConfig(
                name="story",
                prompt="Write about {{ topic }}",
                model_alias="text",
            ),
        ],
        model_configs=[
            dd.ModelConfig(alias="text", model="some-model", provider="some-provider"),
        ],
    )


def _config_without_model_configs() -> dd.DataDesignerConfig:
    return dd.DataDesignerConfig(
        columns=[dd.ExpressionColumnConfig(name="value", expr="'ok'")],
        model_configs=[],
    )


def _make_dd_ctx(
    *,
    validate_errors: list[NDDError] | None = None,
    providers: list[dd.ModelProvider] | None = None,
    providers_exc: BaseException | None = None,
) -> AsyncMock:
    """Stub validation context whose validate / get_model_providers we control.

    Returns the bare :class:`AsyncMock` so
    the test bodies can access the mock-only ``assert_not_called`` /
    ``assert_called_once`` introspection methods directly.
    """
    ctx = AsyncMock()
    ctx.validate.return_value = list(validate_errors or [])
    if providers_exc is not None:
        ctx.get_model_providers.side_effect = providers_exc
    else:
        ctx.get_model_providers.return_value = list(providers or [])
    return ctx


class TestResolveRunnableConfig:
    async def test_clean_config_returns_resolved_values_and_no_errors(self) -> None:
        config = _config_with_aliased_llm_column()
        provider = dd.ModelProvider(name="some-provider", endpoint="http://example.com")
        dd_ctx = _make_dd_ctx(providers=[provider])

        errors, model_configs, model_providers = await resolve_runnable_config(dd_ctx, config)

        assert errors == []
        assert [m.alias for m in model_configs] == ["text"]
        assert model_providers == [provider]

    async def test_context_validate_errors_pass_through(self) -> None:
        config = _config_without_model_configs()
        validate_err = NDDInvalidConfigError("seed thing")
        dd_ctx = _make_dd_ctx(validate_errors=[validate_err])

        errors, model_configs, model_providers = await resolve_runnable_config(dd_ctx, config)

        # Validate errors land in the buffer; model_configs / providers
        # resolution still runs (config has no models, so both are empty).
        assert errors == [validate_err]
        assert model_configs == []
        assert model_providers == []

    async def test_unrecognized_alias_skips_provider_resolution(self) -> None:
        """An unrecognized alias short-circuits provider resolution.

        Without the resolved aliases there's nothing meaningful to resolve
        providers for, so we don't make the SDK call at all.
        """
        config = dd.DataDesignerConfig(
            columns=[
                dd.SamplerColumnConfig(
                    name="topic",
                    sampler_type=dd.SamplerType.CATEGORY,
                    params=dd.CategorySamplerParams(values=["a", "b"]),
                ),
                dd.LLMTextColumnConfig(
                    name="story",
                    prompt="Write about {{ topic }}",
                    model_alias="not-a-real-alias",
                ),
            ],
            # No matching ModelConfig — triggers an NDDInvalidConfigError
            # from get_model_configs.
            model_configs=[],
        )
        dd_ctx = _make_dd_ctx(providers=[])

        errors, model_configs, model_providers = await resolve_runnable_config(dd_ctx, config)

        assert len(errors) == 1
        assert isinstance(errors[0], NDDInvalidConfigError)
        assert "not-a-real-alias" in str(errors[0])
        assert model_configs == []
        assert model_providers == []
        # Provider resolution must be skipped when alias resolution failed.
        dd_ctx.get_model_providers.assert_not_called()

    async def test_provider_resolution_invalid_config_error_is_appended(self) -> None:
        config = _config_with_aliased_llm_column()
        provider_err = NDDInvalidConfigError("Cannot access provider 'some-provider'.")
        dd_ctx = _make_dd_ctx(providers_exc=provider_err)

        errors, model_configs, model_providers = await resolve_runnable_config(dd_ctx, config)

        assert errors == [provider_err]
        # Model configs were resolved (the SDK call below them failed).
        assert [m.alias for m in model_configs] == ["text"]
        # Providers list is empty because resolution failed.
        assert model_providers == []

    async def test_provider_resolution_internal_error_is_appended(self) -> None:
        config = _config_with_aliased_llm_column()
        provider_err = NDDInternalError("transient backend failure")
        dd_ctx = _make_dd_ctx(providers_exc=provider_err)

        errors, model_configs, model_providers = await resolve_runnable_config(dd_ctx, config)

        assert errors == [provider_err]
        assert [m.alias for m in model_configs] == ["text"]
        assert model_providers == []

    async def test_aggregates_context_validate_and_provider_errors(self) -> None:
        """A single pass surfaces every problem that can be detected, not just the first."""
        config = _config_with_aliased_llm_column()
        validate_err = NDDInvalidConfigError("Tool configs are not supported")
        provider_err = NDDInvalidConfigError("Cannot access provider 'some-provider'.")
        dd_ctx = _make_dd_ctx(validate_errors=[validate_err], providers_exc=provider_err)

        errors, _model_configs, _model_providers = await resolve_runnable_config(dd_ctx, config)

        assert errors == [validate_err, provider_err]

    async def test_unexpected_exception_propagates(self) -> None:
        """Truly unexpected errors aren't swallowed — only NDDError-class ones are."""
        config = _config_with_aliased_llm_column()
        dd_ctx = _make_dd_ctx(providers_exc=RuntimeError("oh no"))

        with pytest.raises(RuntimeError, match="oh no"):
            await resolve_runnable_config(dd_ctx, config)
