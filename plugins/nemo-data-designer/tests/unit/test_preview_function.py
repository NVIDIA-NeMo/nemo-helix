# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import data_designer.config as dd
import pytest
from data_designer_nemo.errors import NDDError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_data_designer_plugin.functions import preview as preview_module
from nemo_data_designer_plugin.functions._types import LogFrame, PreviewSpec
from nemo_data_designer_plugin.functions.preview import PreviewFunction
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.dependencies import get_sdk_client, get_sync_sdk_client
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.routes import NDJSON_MEDIA_TYPE, add_function_routes
from pydantic import BaseModel


def _config() -> dd.DataDesignerConfig:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="text", model="model", provider="default/nvidia")]
    )
    builder.add_column(
        column_config=dd.SamplerColumnConfig(
            name="foo",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["a"]),
        )
    )
    return builder.build()


def _sync_sdk() -> NeMoPlatform:
    return NeMoPlatform(base_url="http://testserver", workspace="default")


def _async_sdk() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(base_url="http://testserver", workspace="default")


async def _resolve_runnable_config(
    *_args: object,
) -> tuple[list[NDDError], list[dd.ModelConfig], list[dd.ModelProvider]]:
    return [], [], []


class FakeDataDesigner:
    def check_models(self, _config_builder: dd.DataDesignerConfigBuilder) -> None:
        pass


def _fake_data_designer(*_args: object, **_kwargs: object) -> FakeDataDesigner:
    return FakeDataDesigner()


@pytest.mark.asyncio
async def test_preview_function_streams_worker_frames_and_done(monkeypatch: pytest.MonkeyPatch) -> None:

    def fake_worker(
        send_frame: Callable[[BaseModel], None],
        *_args: object,
    ) -> None:
        send_frame(LogFrame(level="info", message="generated"))

    monkeypatch.setattr(preview_module, "create_data_designer", _fake_data_designer)
    monkeypatch.setattr(preview_module, "resolve_runnable_config", _resolve_runnable_config)
    monkeypatch.setattr(preview_module, "_make_preview_dataset", fake_worker)

    frames = [
        frame
        async for frame in PreviewFunction().run(
            PreviewSpec(config=_config(), num_records=2),
            ctx=FunctionContext(workspace="team-a"),
            sdk=_sync_sdk(),
            async_sdk=_async_sdk(),
        )
    ]

    assert [frame.model_dump()["kind"] for frame in frames] == ["log", "done"]


@pytest.mark.asyncio
async def test_preview_function_runs_model_health_check_off_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    event_loop_thread = threading.current_thread()
    check_models_thread: threading.Thread | None = None

    class RecordingFakeDataDesigner:
        def check_models(self, _config_builder: dd.DataDesignerConfigBuilder) -> None:
            nonlocal check_models_thread
            check_models_thread = threading.current_thread()

    def fake_create_data_designer(*_args: object, **_kwargs: object) -> RecordingFakeDataDesigner:
        return RecordingFakeDataDesigner()

    def fake_worker(*_args: object) -> None:
        pass

    monkeypatch.setattr(preview_module, "create_data_designer", fake_create_data_designer)
    monkeypatch.setattr(preview_module, "resolve_runnable_config", _resolve_runnable_config)
    monkeypatch.setattr(preview_module, "_make_preview_dataset", fake_worker)

    frames = [
        frame
        async for frame in PreviewFunction().run(
            PreviewSpec(config=_config(), num_records=2),
            ctx=FunctionContext(workspace="team-a"),
            sdk=_sync_sdk(),
            async_sdk=_async_sdk(),
        )
    ]

    assert [frame.model_dump()["kind"] for frame in frames] == ["done"]
    assert check_models_thread is not None
    assert check_models_thread is not event_loop_thread


def test_preview_route_streams_ndjson_and_heartbeats(monkeypatch: pytest.MonkeyPatch) -> None:

    def slow_worker(
        send_frame: Callable[[BaseModel], None],
        *_args: object,
    ) -> None:
        send_frame(LogFrame(level="info", message="started"))
        time.sleep(0.05)
        send_frame(LogFrame(level="info", message="more work has been done"))

    monkeypatch.setattr(preview_module, "_make_preview_dataset", slow_worker)
    monkeypatch.setattr(preview_module, "create_data_designer", _fake_data_designer)
    monkeypatch.setattr(preview_module, "resolve_runnable_config", _resolve_runnable_config)

    app = FastAPI()
    app.dependency_overrides[get_sdk_client] = _async_sdk
    app.dependency_overrides[get_sync_sdk_client] = _sync_sdk
    app.include_router(
        add_function_routes(PreviewFunction, heartbeat_interval_seconds=0.01),
        prefix="/apis/data-designer/v2/workspaces/{workspace}",
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/apis/data-designer/v2/workspaces/team-a/preview",
        json=PreviewSpec(config=_config(), num_records=2).model_dump(mode="json"),
    ) as resp:
        assert resp.status_code == 200
        assert NDJSON_MEDIA_TYPE in resp.headers["content-type"]
        frames = [json.loads(line) for line in resp.iter_lines() if line]

    kinds = [frame["kind"] for frame in frames]
    assert "heartbeat" in kinds
    assert kinds[-2:] == ["log", "done"]
