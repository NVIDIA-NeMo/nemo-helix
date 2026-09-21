# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import data_designer.config as dd
import pandas as pd
import pytest
from anonymizer.config.anonymizer_config import AnonymizerInput
from nemo_anonymizer_plugin.app import context as context_module
from nemo_anonymizer_plugin.app.errors import AnonymizerInvalidConfigError
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.model_configs import SelectedModelsOverrides
from nemo_anonymizer_plugin.app.task_config import AnonymizerConfigRequest, RedactRequest
from nemo_anonymizer_plugin.functions import _preview_worker as worker_module
from nemo_anonymizer_plugin.functions._preview_logs import request_callback_cvar
from nemo_anonymizer_plugin.functions.preview import LogFrame, PreviewFunction, PreviewSpec, TraceDatasetFrame
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.function_context import FunctionContext
from pydantic import BaseModel


def _preview_spec() -> PreviewSpec:
    return PreviewSpec(
        config=AnonymizerConfigRequest(replace=RedactRequest()),
        data=AnonymizerInputSpec(source="https://example.com/input.csv", text_column="biography"),
        model_configs=[dd.ModelConfig(alias="detector", model="test/model", provider="provider")],
        num_records=1,
    )


def test_preview_worker_sends_original_text_column_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[BaseModel] = []
    trace_dataframe = pd.DataFrame([{"biography": "hello"}])
    trace_dataframe.attrs["original_text_column"] = "biography"

    class FakeResult:
        dataframe = pd.DataFrame([{"biography": "hello"}])
        failed_records: list[object] = []

        def __init__(self) -> None:
            self.trace_dataframe = trace_dataframe

    class FakeAnonymizer:
        def preview(self, **kwargs: object) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(worker_module, "_make_anonymizer", lambda **kwargs: FakeAnonymizer())

    worker_module._make_preview(
        frames.append,
        _preview_spec(),
        data=AnonymizerInput(source="https://example.com/input.csv", text_column="biography"),
        model_configs_yaml="model_configs:\n- alias: detector\n  model: test/model\n  provider: provider\n",
        dd_providers=None,
        num_records=1,
    )

    trace_frame = next(frame for frame in frames if isinstance(frame, TraceDatasetFrame))
    assert trace_frame.original_text_column == "biography"


def test_preview_worker_requires_model_configs() -> None:
    with pytest.raises(RuntimeError, match="requires resolved model_configs"):
        worker_module._make_anonymizer(model_configs_yaml="", dd_providers=None)


@pytest.mark.asyncio
async def test_preview_function_resets_request_log_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_worker(
        send_frame: Callable[[BaseModel], None],
        *args: object,
    ) -> None:
        send_frame(LogFrame(level="info", message="generated"))

    igw_lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(context_module, "make_model_provider_registry", igw_lookup)
    monkeypatch.setattr(worker_module, "_make_preview", fake_worker)
    async_sdk = AsyncMock(spec=AsyncNeMoPlatform)

    frames = [
        frame
        async for frame in PreviewFunction().run(
            _preview_spec(),
            ctx=FunctionContext(workspace="team-a"),
            async_sdk=async_sdk,
        )
    ]

    igw_lookup.assert_awaited_once()
    assert igw_lookup.await_args is not None
    assert igw_lookup.await_args.kwargs["sdk"] is async_sdk
    assert [frame.model_dump()["kind"] for frame in frames] == ["log", "done"]
    assert request_callback_cvar.get() is None


@pytest.mark.asyncio
async def test_preview_function_rejects_selected_models_without_model_configs() -> None:
    spec = _preview_spec().model_copy(
        update={
            "model_configs": None,
            "selected_models": SelectedModelsOverrides(detection={"entity_detector": "detector"}),
        }
    )

    with pytest.raises(AnonymizerInvalidConfigError, match="selected_models requires model_configs"):
        [
            frame
            async for frame in PreviewFunction().run(
                spec,
                ctx=FunctionContext(workspace="team-a"),
                async_sdk=AsyncMock(spec=AsyncNeMoPlatform),
            )
        ]


@pytest.mark.asyncio
async def test_preview_submit_requires_model_configs() -> None:
    with pytest.raises(AnonymizerInvalidConfigError, match="model_configs are required"):
        [
            frame
            async for frame in PreviewFunction().run(
                _preview_spec().model_copy(update={"model_configs": None}),
                ctx=FunctionContext(workspace="team-a"),
                async_sdk=AsyncMock(spec=AsyncNeMoPlatform),
            )
        ]
