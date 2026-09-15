# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared compile/run helpers for retrieval jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_data_designer_plugin.config import get_config
from nemo_data_designer_plugin.retrieval.corpus import HF_TOKEN_ENVVAR
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
    GPUExecutionProviderSpec,
    PlatformJobStep,
)
from nemo_platform_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.image import get_qualified_image
from nmp.customization_common.schemas.file_io import DownloadItem, FileIOTaskConfig, FileSetRef
from pydantic import BaseModel

_ENTRYPOINT = ["python", "-m"]
RETRIEVAL_MINE_MODULE = "nmp.automodel.tasks.retrieval_mine"
_FILE_IO_MODULE = "nmp.customization_common.tasks.file_io"
_FILE_IO_ARGS = ["--service-source", "automodel", "--service-name", "customizer"]


def _persistent_storage_environment() -> list[EnvironmentVariable]:
    return [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]


def _hf_token_environment(hf_token_secret: str | None) -> list[EnvironmentVariable]:
    """Project a ``hf_token_secret`` reference into the step env, never its plaintext.

    The jobs service resolves ``from_secret`` against the job workspace, so an
    unqualified secret name works the same way it does for every other job.
    """
    if not hf_token_secret:
        return []
    return [
        EnvironmentVariable(
            name=HF_TOKEN_ENVVAR,
            from_secret=EnvironmentVariableFromSecret(name=hf_token_secret),
        )
    ]


def cpu_retrieval_step(
    name: str,
    module: str,
    spec: BaseModel,
    profile: str | None,
    module_args: list[str] | None = None,
    image: str = "nmp-cpu-tasks",
    hf_token_secret: str | None = None,
) -> PlatformJobStep:
    return PlatformJobStep(
        name=name,
        executor=CPUExecutionProviderSpec(
            profile=profile or get_config().job_executor_profile,
            provider="cpu",
            container=ContainerSpec(
                image=get_qualified_image(image),
                entrypoint=_ENTRYPOINT,
                command=[module, *(module_args or [])],
            ),
        ),
        config=spec.model_dump(mode="json"),
        environment=[*_persistent_storage_environment(), *_hf_token_environment(hf_token_secret)],
    )


def gpu_retrieval_step(name: str, module: str, spec: BaseModel, profile: str | None) -> PlatformJobStep:
    """GPU step in the automodel training image.

    Falls back to the plugin's ``job_executor_profile``. The profile must name a
    registered GPU executor: the jobs API rejects a spec whose (provider, profile) is unknown.
    """
    return PlatformJobStep(
        name=name,
        executor=GPUExecutionProviderSpec(
            profile=profile or get_config().job_executor_profile,
            provider="gpu",
            container=ContainerSpec(
                image=get_qualified_image("nmp-automodel-training"),
                entrypoint=_ENTRYPOINT,
                command=[module],
            ),
        ),
        config=spec.model_dump(mode="json"),
        environment=[
            *_persistent_storage_environment(),
            EnvironmentVariable(name="HF_HUB_OFFLINE", value="1"),
            EnvironmentVariable(name="TRANSFORMERS_OFFLINE", value="1"),
        ],
    )


async def retrieval_step(
    name: str,
    module: str,
    spec: BaseModel,
    profile: str | None,
    async_sdk: object,
    gpu: bool = False,
    hf_token_secret: str | None = None,
) -> PlatformJobStep:
    del async_sdk
    if gpu:
        # The GPU mining step runs with ``HF_HUB_OFFLINE``; it never reaches the Hub.
        return gpu_retrieval_step(name, module, spec, profile)
    return cpu_retrieval_step(name, module, spec, profile, hf_token_secret=hf_token_secret)


async def model_download_step(
    fileset: str,
    profile: str | None,
    async_sdk: object,
) -> PlatformJobStep:
    """Download a model fileset into the job's shared ``model`` directory."""
    del async_sdk
    config = FileIOTaskConfig(download=[DownloadItem(src=FileSetRef.model_validate(fileset), dest="model")])
    return cpu_retrieval_step(
        "retrieval-model-download",
        _FILE_IO_MODULE,
        config,
        profile,
        module_args=_FILE_IO_ARGS,
        image="nmp-customizer-tasks",
    )


def work_dir(ctx: Any, name: str) -> Path:
    base = ctx.storage.persistent or ctx.storage.ephemeral
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
