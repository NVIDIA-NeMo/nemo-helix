# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AutomodelJob.compile error mapping.

``validate_for_training`` rejects inconsistent parallelism/batch topologies.
Those are user input errors, so they have to leave ``compile`` as
``PlatformJobCompilationError`` — the api_factory only maps that type to a
422, and anything else escapes as a 500.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nemo_automodel_plugin.schema import AutomodelJobOutput
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError


def _make_canonical(**parallelism: Any) -> AutomodelJobOutput:
    return AutomodelJobOutput.model_validate(
        {
            "model": "default/base",
            "dataset": {"training": "default/train"},
            "training": {"training_type": "sft"},
            "schedule": {"epochs": 1},
            "batch": {"global_batch_size": 4, "micro_batch_size": 1},
            "optimizer": {},
            "parallelism": {"num_nodes": 2, "num_gpus_per_node": 8, **parallelism},
            "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
        },
    )


def _compile(canonical: AutomodelJobOutput) -> Any:
    async def run_compile() -> Any:
        async_sdk = AsyncNeMoPlatform(
            base_url="http://test",
            workspace="default",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
            ),
        )
        try:
            return await AutomodelJob.compile(
                workspace="default",
                spec=canonical,
                entity_client=object(),
                job_name=None,
                async_sdk=async_sdk,
            )
        finally:
            await async_sdk.close()

    return asyncio.run(run_compile())


class TestCompileValidationErrors:
    def test_indivisible_batch_raises_compilation_error(self) -> None:
        # global_batch_size=4 with data_parallel_size=16 (2 nodes x 8 GPUs, TP=1).
        canonical = _make_canonical(tensor_parallel_size=1)
        with patch("nemo_automodel_plugin.jobs.jobs.require_container_runtime"):
            with pytest.raises(PlatformJobCompilationError, match="global_batch_size"):
                _compile(canonical)

    def test_indivisible_model_parallel_raises_compilation_error(self) -> None:
        # 16 total GPUs is not divisible by tensor_parallel_size=5.
        canonical = _make_canonical(tensor_parallel_size=5)
        with patch("nemo_automodel_plugin.jobs.jobs.require_container_runtime"):
            with pytest.raises(PlatformJobCompilationError, match="Total GPUs"):
                _compile(canonical)
