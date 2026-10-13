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
from nemo_anonymizer_plugin.functions import preview as preview_module
from nemo_anonymizer_plugin.functions._preview_logs import request_callback_cvar
from nemo_anonymizer_plugin.functions.preview import LogFrame, PreviewFunction, PreviewSpec, TraceDatasetFrame
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.function_context import FunctionContext
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


def test_make_anonymizer_uses_gateway_when_caller_supplies_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the caller supplies its own detector, the worker must build the plain
    # gateway Anonymizer and must NOT spin up the in-process GLiNER runtime.
    sentinel = object()
    monkeypatch.setattr(worker_module, "Anonymizer", lambda **kwargs: sentinel)

    def _must_not_run(**kwargs: object) -> object:
        raise AssertionError("in-process GLiNER must not be built on the gateway path")

    monkeypatch.setattr(worker_module, "build_gliner_anonymizer", _must_not_run)

    result = worker_module._make_anonymizer(
        model_configs_yaml="model_configs: []",
        dd_providers=None,
        use_in_process_detector=False,
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_preview_function_resets_request_log_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_worker(
        send_frame: Callable[[BaseModel], None],
        *args: object,
        **kwargs: object,
    ) -> None:
        send_frame(LogFrame(level="info", message="generated"))

    igw_lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(context_module, "make_model_provider_registry", igw_lookup)
    monkeypatch.setattr(worker_module, "_make_preview", fake_worker)
    # The GLiNER detector weights are ensured out-of-band; stub readiness so the
    # test stays focused on preview frame/callback behavior.
    monkeypatch.setattr(preview_module, "ensure_gliner_fileset_async", AsyncMock(return_value=None))
    monkeypatch.setattr(preview_module, "is_gliner_cached", lambda: True)
    async_sdk = AsyncMock(spec=AsyncNemoClient)

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
    assert igw_lookup.await_args.kwargs["client"] is async_sdk
    assert [frame.model_dump()["kind"] for frame in frames] == ["log", "done"]
    assert request_callback_cvar.get() is None


@pytest.mark.asyncio
async def test_preview_function_emits_error_frame_when_model_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    igw_lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(context_module, "make_model_provider_registry", igw_lookup)
    monkeypatch.setattr(preview_module, "ensure_gliner_fileset_async", AsyncMock(return_value=None))
    # Cold cache + a failing download: the stream must surface an Error frame, not die silently.
    monkeypatch.setattr(preview_module, "is_gliner_cached", lambda: False)

    def _boom(_base_url: str) -> None:
        raise RuntimeError("files unreachable")

    monkeypatch.setattr(preview_module, "prewarm_gliner_cache", _boom)

    frames = [
        frame
        async for frame in PreviewFunction().run(
            _preview_spec(),
            ctx=FunctionContext(workspace="team-a"),
            async_sdk=AsyncMock(spec=AsyncNemoClient),
        )
    ]

    # A started model_download frame, an error log, then the Error frame. There is no
    # duplicate info LogFrame for the started message (callers get it from the
    # model_download frame itself).
    kinds = [frame.model_dump()["kind"] for frame in frames]
    assert kinds == ["model_download", "log", "error"]
    # The raw exception text (which may carry a path/connection detail) must NOT leak
    # into the client-facing frames; a fixed message + reason code is sent instead.
    error_frame = frames[-1].model_dump()
    assert "files unreachable" not in error_frame["message"]
    assert error_frame["message"] == "Failed to download the PII detector model."
    assert error_frame["details"]["type"] == "ModelDownloadError"


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
                async_sdk=AsyncMock(spec=AsyncNemoClient),
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
                async_sdk=AsyncMock(spec=AsyncNemoClient),
            )
        ]
