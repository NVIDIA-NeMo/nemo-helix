# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import data_designer.config as dd
import pytest
from data_designer_nemo.errors import NDDInternalError
from data_designer_nemo.person_sampling import (
    ensure_nemotron_personas_filesets,
)
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NotFoundError, PermissionDeniedError


def _make_person_sampler_column(name: str, locale: str) -> dd.SamplerColumnConfig:
    return dd.SamplerColumnConfig(
        name=name,
        sampler_type=dd.SamplerType.PERSON,
        params=dd.PersonSamplerParams(locale=locale),
    )


def _make_config(*columns: dd.SamplerColumnConfig) -> dd.DataDesignerConfig:
    builder = dd.DataDesignerConfigBuilder()
    for column in columns:
        builder.add_column(column_config=column)
    return builder.build()


def _mock_http_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"detail": "error"}
    resp.text = "error"
    return resp


@pytest.mark.asyncio
async def test_ensure_nemotron_personas_filesets_checks_each_locale() -> None:
    sdk = AsyncMock(spec=AsyncNemoClient)
    mock_files = MagicMock()
    mock_files.get_fileset = AsyncMock()
    config = _make_config(
        _make_person_sampler_column("person_us", "en_US"),
        _make_person_sampler_column("person_jp", "ja_JP"),
    )

    with patch("data_designer_nemo.person_sampling.client_from_platform", return_value=mock_files):
        await ensure_nemotron_personas_filesets(config, sdk)

    assert mock_files.get_fileset.await_count == 2


@pytest.mark.asyncio
async def test_ensure_nemotron_personas_filesets_raises_error_for_missing_fileset() -> None:
    sdk = AsyncMock(spec=AsyncNemoClient)
    mock_files = MagicMock()
    mock_files.get_fileset = AsyncMock(side_effect=NotFoundError(_mock_http_response(404)))
    config = _make_config(_make_person_sampler_column("person", "en_US"))

    with patch("data_designer_nemo.person_sampling.client_from_platform", return_value=mock_files):
        with pytest.raises(NDDInternalError):
            await ensure_nemotron_personas_filesets(config, sdk)


@pytest.mark.asyncio
async def test_ensure_nemotron_personas_filesets_raises_error_for_permission_error() -> None:
    sdk = AsyncMock(spec=AsyncNemoClient)
    mock_files = MagicMock()
    mock_files.get_fileset = AsyncMock(side_effect=PermissionDeniedError(_mock_http_response(403)))
    config = _make_config(_make_person_sampler_column("person", "en_US"))

    with patch("data_designer_nemo.person_sampling.client_from_platform", return_value=mock_files):
        with pytest.raises(NDDInternalError):
            await ensure_nemotron_personas_filesets(config, sdk)


@pytest.mark.asyncio
async def test_ensure_nemotron_personas_filesets_raises_internal_error_on_other_errors() -> None:
    sdk = AsyncMock(spec=AsyncNemoClient)
    mock_files = MagicMock()
    mock_files.get_fileset = AsyncMock(side_effect=RuntimeError("something went wrong"))
    config = _make_config(_make_person_sampler_column("person", "en_US"))

    with patch("data_designer_nemo.person_sampling.client_from_platform", return_value=mock_files):
        with pytest.raises(NDDInternalError):
            await ensure_nemotron_personas_filesets(config, sdk)
