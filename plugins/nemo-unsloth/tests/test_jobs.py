# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for UnslothJob lifecycle (to_spec + compile).

After the 2026 migration from local run to container submit we no longer
exercise ``train_sft`` from these tests — that lives in the
``nmp-unsloth-training`` container's smoke test. Here we just pin:

- ``to_spec`` resolves output naming + fileset against a stub SDK.
- ``compile`` delegates to the service-side compiler (we patch it out)
  and returns the resulting ``PlatformJobSpec`` after the container
  runtime check.
- The container runtime check fires when the platform has no usable
  container runtime (neither Kubernetes nor Docker).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_unsloth_plugin.jobs.jobs import UnslothJob
from nemo_unsloth_plugin.schema import UnslothJobInput
from nmp.unsloth.schemas import UnslothJobOutput

BASE_URL = "http://test"


def _input_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": {"name": "default/base"},
        "dataset": {"path": "default/training"},
        "schedule": {"max_steps": 60},
    }
    base.update(overrides)
    return base


def _model_json() -> dict[str, object]:
    return {
        "id": "model-base",
        "name": "base",
        "workspace": "default",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
        "spec": None,
        "fileset": "default/base-fs",
        "trust_remote_code": False,
    }


def _fileset_json(workspace: str, name: str) -> dict[str, object]:
    return {
        "id": f"{workspace}-{name}",
        "name": name,
        "workspace": workspace,
        "description": "",
        "purpose": "generic",
        "storage": {"type": "local", "path": "/tmp/files"},
        "metadata": {},
        "custom_fields": {},
        "project": "",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }


async def _make_canonical_async(workspace: str = "default", **overrides: Any) -> UnslothJobOutput:
    spec = UnslothJobInput.model_validate(_input_dict(**overrides))

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        parts = path.split("/")
        if path.startswith("/apis/models/v2/workspaces/"):
            return httpx.Response(200, request=request, json=_model_json())
        if path.startswith("/apis/files/v2/workspaces/"):
            return httpx.Response(200, request=request, json=_fileset_json(parts[5], parts[7]))
        return httpx.Response(404, request=request, json={"detail": "unexpected request"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async_sdk = AsyncNeMoPlatform(base_url=BASE_URL, workspace="default", http_client=http_client)
    try:
        output = await UnslothJob.to_spec(
            spec,
            workspace=workspace,
            entity_client=object(),
            async_sdk=async_sdk,
            is_local=False,
        )
        assert isinstance(output, UnslothJobOutput)
        return output
    finally:
        await async_sdk.close()


def _make_canonical(workspace: str = "default", **overrides: Any) -> UnslothJobOutput:
    return asyncio.run(_make_canonical_async(workspace, **overrides))


def _compile_sdk() -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
        ),
    )


class TestToSpec:
    def test_to_spec_resolves_output(self) -> None:
        out = _make_canonical()
        assert isinstance(out, UnslothJobOutput)
        assert out.output.type == "adapter"
        assert out.output.save_method == "lora"
        # Fileset defaults to the entity name (mirrors automodel).
        assert out.output.fileset == out.output.name


class TestCompile:
    def test_compile_delegates_to_service_compiler(self) -> None:
        """When the runtime check passes, ``compile`` returns whatever the service builds."""
        canonical = _make_canonical()
        async_sdk = _compile_sdk()
        fake_spec = SimpleNamespace(
            steps=["model-and-dataset-download", "training", "model-upload", "model-entity-creation"]
        )

        try:
            with (
                patch("nemo_unsloth_plugin.jobs.jobs.require_container_runtime"),
                patch(
                    "nemo_unsloth_plugin.jobs.jobs.platform_job_config_compiler",
                    new=AsyncMock(return_value=fake_spec),
                ) as compile_mock,
                patch(
                    "nemo_unsloth_plugin.jobs.jobs.validate_gpu_available_for_docker",
                    new=MagicMock(),
                ) as validate_mock,
            ):
                result = asyncio.run(
                    UnslothJob.compile(
                        workspace="default",
                        spec=canonical,
                        entity_client=object(),
                        job_name="my-unsloth-job",
                        async_sdk=async_sdk,
                        profile=None,
                    ),
                )
        finally:
            asyncio.run(async_sdk.close())

        assert result is fake_spec
        compile_mock.assert_awaited_once()
        validate_mock.assert_called_once_with(fake_spec)
        kwargs = compile_mock.await_args.kwargs
        assert kwargs["workspace"] == "default"
        assert kwargs["job_name"] == "my-unsloth-job"
        # Profile falls through to the unsloth config default (`gpu`).
        assert kwargs["profile"] == "gpu"

    def test_compile_passes_caller_profile_override(self) -> None:
        canonical = _make_canonical()
        async_sdk = _compile_sdk()
        try:
            with (
                patch("nemo_unsloth_plugin.jobs.jobs.require_container_runtime"),
                patch(
                    "nemo_unsloth_plugin.jobs.jobs.platform_job_config_compiler",
                    new=AsyncMock(return_value=SimpleNamespace(steps=[])),
                ) as compile_mock,
                patch("nemo_unsloth_plugin.jobs.jobs.validate_gpu_available_for_docker"),
            ):
                asyncio.run(
                    UnslothJob.compile(
                        workspace="default",
                        spec=canonical,
                        entity_client=object(),
                        job_name=None,
                        async_sdk=async_sdk,
                        profile="gpu_distributed",
                    ),
                )
        finally:
            asyncio.run(async_sdk.close())

        await_args = compile_mock.await_args
        assert await_args is not None
        assert await_args.kwargs["profile"] == "gpu_distributed"

    def test_compile_rejects_runtime_without_container_support(self) -> None:
        canonical = _make_canonical()
        # Force the runtime check to raise so we don't need a real runtime
        # in CI. The check is what runs first; the rest never executes.
        async_sdk = _compile_sdk()
        with patch(
            "nemo_unsloth_plugin.jobs.jobs.require_container_runtime",
            side_effect=PlatformJobCompilationError("no container runtime"),
        ):
            try:
                with pytest.raises(PlatformJobCompilationError, match="no container runtime"):
                    asyncio.run(
                        UnslothJob.compile(
                            workspace="default",
                            spec=canonical,
                            entity_client=object(),
                            job_name=None,
                            async_sdk=async_sdk,
                        ),
                    )
            finally:
                asyncio.run(async_sdk.close())


class TestNoRun:
    def test_unsloth_job_is_abstract_because_run_is_not_implemented(self) -> None:
        """``NemoJob.run`` is ``@abstractmethod`` and we deliberately don't override it.

        Pin so a future override doesn't silently re-enable local run —
        Unsloth migrated to container submit in 2026. ``run`` lives in
        the ``nmp-unsloth-training`` container's ``__main__`` now.
        """
        with pytest.raises(TypeError, match="abstract"):
            UnslothJob()
