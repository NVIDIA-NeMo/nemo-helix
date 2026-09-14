# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anonymizer task — runs inside the nmp-cpu-tasks container."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from anonymizer.interface.anonymizer import Anonymizer
from data_designer.config.models import ModelProvider as DDModelProvider
from data_designer_nemo.model_provider import (
    get_nmp_provider,
    parse_provider_reference,
)
from nemo_anonymizer_plugin.app.input import prepare_anonymizer_input
from nemo_anonymizer_plugin.app.task_config import AnonymizerStepConfig
from nemo_anonymizer_plugin.app.upstream_logging import preserve_root_logging
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import PlatformJobResults
from nemo_platform_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.sdk_provider import get_platform_sdk

logger = logging.getLogger(__name__)

ARTIFACTS_RESULT_NAME = "artifacts"
_TASK_LOG_HANDLER_MARKER = "_nemo_anonymizer_task_handler"


def run(sdk: NeMoPlatform | None = None) -> int:
    try:
        service_sdk = sdk or get_platform_sdk(as_service="anonymizer")
        return run_step_config(_load_step_config(), ctx=_get_ctx(service_sdk), sdk=service_sdk)
    except Exception:
        logger.exception("Anonymizer task failed")
        return 1


def run_step_config(
    step_config: AnonymizerStepConfig,
    *,
    ctx: JobContext,
    sdk: NeMoPlatform | None = None,
) -> int:
    try:
        return _run_with_step_config(sdk, step_config, ctx=ctx)
    except Exception:
        logger.exception("Anonymizer task failed")
        return 1


def _run_with_step_config(
    service_sdk: NeMoPlatform | None,
    step_config: AnonymizerStepConfig,
    *,
    ctx: JobContext,
) -> int:
    _configure_logging()
    if service_sdk is None:
        raise RuntimeError("Remote anonymizer task requires a NeMo Platform SDK.")

    storage_path = ctx.storage.persistent
    workspace = ctx.workspace

    request = step_config.request
    if not step_config.model_configs_yaml:
        raise RuntimeError("Remote anonymizer task requires resolved model_configs.")
    dd_providers = _resolve_provider_endpoints(service_sdk, step_config, workspace)
    prepared_input = prepare_anonymizer_input(
        request.data,
        sdk=service_sdk,
        workspace=workspace,
        allow_local_paths=False,
    )

    try:
        with preserve_root_logging():
            anonymizer = Anonymizer(
                model_configs=step_config.model_configs_yaml,
                model_providers=dd_providers,
                artifact_path=storage_path / "anonymizer-artifacts",
            )
        logger.info("Running anonymizer pipeline")
        result = anonymizer.run(config=request.config, data=prepared_input.input)
    finally:
        prepared_input.cleanup()

    artifacts_dir = storage_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_parquet(artifacts_dir / "dataset.parquet", index=False)
    result.trace_dataframe.to_parquet(artifacts_dir / "trace.parquet", index=False)
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"original_text_column": _get_original_text_column(result.trace_dataframe, request.data.text_column)},
            f,
        )
    if result.failed_records:
        with open(artifacts_dir / "failed_records.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "record_id": getattr(r, "record_id", None),
                        "step": getattr(r, "step", None),
                        "reason": getattr(r, "reason", None),
                    }
                    for r in result.failed_records
                ],
                f,
                indent=2,
            )

    ctx.results.save(ARTIFACTS_RESULT_NAME, artifacts_dir)

    logger.info(
        "Anonymizer task complete (records=%d, failures=%d)",
        len(result.dataframe),
        len(result.failed_records or []),
    )
    return 0


def _resolve_provider_endpoints(
    sdk: NeMoPlatform,
    step_config: AnonymizerStepConfig,
    workspace: str,
) -> list[DDModelProvider] | None:
    """Re-resolve provider endpoints in the task environment.

    The ``ModelProvider.endpoint`` URL captured at API time may resolve to a
    different in-cluster URL inside the task pod (matches what DD does).
    """
    if not step_config.dd_model_providers:
        return None
    models = client_from_platform(sdk, ModelsClient)
    refreshed: list[DDModelProvider] = []
    for raw in step_config.dd_model_providers:
        provider = DDModelProvider.model_validate(raw)
        provider_workspace, provider_name = parse_provider_reference(provider.name, workspace)
        nmp_provider = get_nmp_provider(sdk, provider_workspace, provider_name)
        provider.endpoint = models.get_provider_route_openai_url(nmp_provider)
        refreshed.append(provider)
    return refreshed


def _configure_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for module in ("anonymizer", "data_designer", "nemo_anonymizer_plugin"):
        module_logger = logging.getLogger(module)
        handler = _get_task_log_handler(module_logger)
        if handler is None:
            handler = logging.StreamHandler()
            setattr(handler, _TASK_LOG_HANDLER_MARKER, True)
            module_logger.addHandler(handler)
        handler.setFormatter(formatter)
        module_logger.setLevel(logging.INFO)
        module_logger.propagate = False


def _get_task_log_handler(module_logger: logging.Logger) -> logging.Handler | None:
    for handler in module_logger.handlers:
        if getattr(handler, _TASK_LOG_HANDLER_MARKER, False):
            return handler
    return None


def _get_original_text_column(trace_dataframe: object, fallback: str) -> str:
    attrs = getattr(trace_dataframe, "attrs", {})
    if isinstance(attrs, dict):
        value = attrs.get("original_text_column")
        if isinstance(value, str) and value:
            return value
    return fallback


def _load_step_config() -> AnonymizerStepConfig:
    with open(_get_required_env(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR), "r") as f:
        return AnonymizerStepConfig.model_validate_json(f.read())


def _get_ctx(sdk: NeMoPlatform) -> JobContext:
    workspace = _get_workspace()
    job_name = _get_job_name()
    storage = StoragePaths(
        ephemeral=Path(_get_required_env(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR)),
        persistent=Path(_get_required_env(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)),
    )
    results = PlatformJobResults(
        workspace=workspace,
        job_name=job_name,
        client=client_from_platform(sdk, NemoClient),
    )
    return JobContext(
        workspace=workspace,
        job_id=job_name,
        storage=storage,
        results=results,
    )


def _get_job_name() -> str:
    return _get_required_env(NEMO_JOB_ID_ENVVAR)


def _get_workspace() -> str:
    return _get_required_env(NEMO_JOB_WORKSPACE_ENVVAR)


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required job environment variable: {name}")
    return value
