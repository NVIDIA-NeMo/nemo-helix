# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch task entry point for the safe-synthesizer service.

This module serves as the entry point for the safe-synthesizer batch task,
which generates synthetic data using the Nemo Safe Synthesizer SDK.
"""

import os

# Disable PyTorch inductor remote cache to avoid Redis warnings from vLLM.
# These must be set before any PyTorch imports.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", "0")

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from filesets import FilesetFileSystem, parse_fileset_ref
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.config import get_platform_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import (
    DEFAULT_TASK_STORAGE_PATH,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from nemo_helix_plugin.jobs.file_manager import FilesetFileManager, TmpDirPath
from nemo_helix_plugin.jobs.schemas import FileStorageType, HelixJobResultCreateRequest
from nemo_helix_plugin.sdk_provider import get_platform_sdk
from nemo_safe_synthesizer.config.internal_results import SafeSynthesizerResults
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.observability import initialize_observability
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer_plugin.job_config import (
    SafeSynthesizerJobConfig,
    parse_pretrained_model_job_ref,
)
from nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.adapter_resolution import (
    embed_run_config_in_adapter,
    is_adapter_reuse_requested,
    run_generation_from_prior_adapter,
)
from nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.jsonl_loader import load_jsonl_file
from nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.logging_setup import configure_logging
from nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.model_init import init_models_sync

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))

logger = logging.getLogger("safe_synthesizer")


def download_from_fileset(fileset_url: str) -> pd.DataFrame:
    """Download a dataset from a fileset and load it as a DataFrame."""
    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, "default")
    workspace, fileset_name, _ = parse_fileset_ref(fileset_url, workspace_fallback=workspace)
    sdk = get_platform_sdk()

    file_manager = FilesetFileManager(
        workspace=workspace,
        fileset_name=fileset_name,
        filesystem=FilesetFileSystem(client=client_from_platform(sdk, FilesClient)),
        ensure_fileset_exists=False,
    )

    logger.info("Downloading dataset from fileset: %s", fileset_url)
    tmp_dir_path = file_manager.download_from_url(fileset_url)
    local_path = tmp_dir_path.path

    try:
        if not local_path.exists():
            raise FileNotFoundError(f"Failed to download file from fileset. File not found at: {local_path}")
        if local_path.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {local_path}")
        return _load_file_as_dataframe(local_path)
    finally:
        tmp_dir_path.cleanup_tmp_dir()


def _load_file_as_dataframe(local_path: Path) -> pd.DataFrame:
    """Load a local file as a DataFrame based on extension."""
    suffix = local_path.suffix.lower()
    if suffix == ".csv":
        logger.info("Loading CSV file: %s", local_path)
        return pd.read_csv(local_path)
    elif suffix == ".parquet":
        logger.info("Loading Parquet file: %s", local_path)
        return pd.read_parquet(local_path)
    elif suffix == ".jsonl":
        logger.info("Loading JSONL file: %s", local_path)
        return load_jsonl_file(str(local_path))
    elif suffix == ".json":
        logger.info("Loading JSON file: %s", local_path)
        return pd.read_json(local_path)
    return _load_local_dataset(str(local_path))


def _load_local_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a local path using JSONL repair or HuggingFace datasets."""
    jsonl_files = []
    other_files = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                if filename.endswith(".jsonl"):
                    jsonl_files.append(filepath)
                elif filename.endswith((".csv", ".parquet", ".json")):
                    other_files.append(filepath)
    elif os.path.isfile(path):
        if path.endswith(".jsonl"):
            jsonl_files.append(path)
        else:
            other_files.append(path)

    if jsonl_files:
        if len(jsonl_files) > 1:
            logger.warning("Multiple JSONL files found (%d), using the first one", len(jsonl_files))
        return load_jsonl_file(jsonl_files[0])

    hf_dataset = load_dataset(path=path)
    if isinstance(hf_dataset, DatasetDict):
        if len(hf_dataset) > 1:
            logger.warning("Multiple datasets found (%d), using the first one", len(hf_dataset))
        hf_dataset = hf_dataset[list(hf_dataset.keys())[0]]
    if not isinstance(hf_dataset, Dataset):
        raise ValueError(f"Expected a dataset at {path!r}, got {type(hf_dataset).__name__}")
    return cast(pd.DataFrame, hf_dataset.to_pandas())


def upload_results(result: SafeSynthesizerResults, adapter_path: Path | None = None):
    """Upload job results to the files service using FilesetFileManager."""
    job_id = os.environ.get(NEMO_JOB_ID_ENVVAR)
    if not job_id:
        raise ValueError(f"{NEMO_JOB_ID_ENVVAR} is not set")

    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, "default")
    sdk = get_platform_sdk()
    fileset_name = f"job-results-{job_id}"
    file_manager = FilesetFileManager(
        workspace=workspace,
        fileset_name=fileset_name,
        filesystem=FilesetFileSystem(client=client_from_platform(sdk, FilesClient)),
        ensure_fileset_exists=True,
    )
    file_manager.validate_storage()

    job = client_from_platform(sdk, JobsClient).get_job(name=job_id, workspace=workspace).data()
    attempt_id = job.attempt_id

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if result.synthetic_data is None:
            raise ValueError("Safe Synthesizer did not produce synthetic data")
        result_csv_path = temp_path / "result.csv"
        result.synthetic_data.to_csv(result_csv_path, index=False)
        artifact_url = file_manager.upload(result_csv_path, f"results/{attempt_id}/synthetic-data")
        _create_job_result(sdk, workspace, job_id, "synthetic-data", artifact_url)

        summary_json_path = temp_path / "summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(result.summary.model_dump(), f)
        artifact_url = file_manager.upload(summary_json_path, f"results/{attempt_id}/summary")
        _create_job_result(sdk, workspace, job_id, "summary", artifact_url)

        if result.evaluation_report_html:
            report_html_path = temp_path / "report.html"
            with open(report_html_path, "w", encoding="utf-8") as f:
                f.write(result.evaluation_report_html)
            artifact_url = file_manager.upload(report_html_path, f"results/{attempt_id}/evaluation-report")
            _create_job_result(sdk, workspace, job_id, "evaluation-report", artifact_url)

    if adapter_path is not None and adapter_path.exists():
        embed_run_config_in_adapter(adapter_path)
        artifact_url = file_manager.upload(adapter_path, f"results/{attempt_id}/adapter")
        _create_job_result(sdk, workspace, job_id, "adapter", artifact_url)


def _create_job_result(sdk: NeMoHelix, workspace: str, job_name: str, result_name: str, artifact_url: str):
    """Create a job result record."""
    client_from_platform(sdk, JobsClient).create_job_result(
        name=result_name,
        job=job_name,
        workspace=workspace,
        body=HelixJobResultCreateRequest(artifact_url=artifact_url, artifact_storage_type=FileStorageType.FILESET),
    )
    logger.info("Created job result: %s", result_name)


def _resolve_pretrained_model(
    job_config: SafeSynthesizerJobConfig,
    *,
    workspace: str,
    sdk: NeMoHelix | None = None,
) -> tuple[TmpDirPath | None, Path | None]:
    """Download a prior job's adapter artifact from Files when ``pretrained_model_job`` is set."""
    if not job_config.pretrained_model_job:
        return None, None

    if sdk is None:
        sdk = get_platform_sdk()

    model_workspace, model_job = parse_pretrained_model_job_ref(
        job_config.pretrained_model_job,
        workspace_fallback=workspace,
    )
    try:
        adapter_result = (
            client_from_platform(sdk, JobsClient)
            .get_job_result(name="adapter", job=model_job, workspace=model_workspace)
            .data()
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve adapter result for pretrained_model_job={job_config.pretrained_model_job!r}"
        ) from e

    fileset_workspace, fileset_name, _ = parse_fileset_ref(
        adapter_result.artifact_url,
        workspace_fallback=model_workspace,
    )
    file_manager = FilesetFileManager(
        workspace=fileset_workspace,
        fileset_name=fileset_name,
        filesystem=FilesetFileSystem(client=client_from_platform(sdk, FilesClient)),
        ensure_fileset_exists=False,
    )
    tmp_dir_path = file_manager.download_from_url(adapter_result.artifact_url)
    adapter_path = tmp_dir_path.path
    logger.info(
        "Resolved pretrained_model_job %s to adapter artifact at %s",
        job_config.pretrained_model_job,
        adapter_path,
    )
    return tmp_dir_path, adapter_path


def _setup_classify_endpoint():
    """Set up upstream Safe Synthesizer PII classification env vars from platform env vars."""
    endpoint_path = os.environ.get("CLASSIFY_LLM_ENDPOINT_PATH")
    if endpoint_path:
        models_url = os.environ.get("NHX_MODELS_URL")
        if not models_url:
            logger.warning(
                "CLASSIFY_LLM_ENDPOINT_PATH is set but NHX_MODELS_URL is not available. "
                "Column classification may not work correctly."
            )
            return
        full_url = models_url.rstrip("/") + endpoint_path
        os.environ["NSS_INFERENCE_ENDPOINT"] = full_url
        os.environ.setdefault("NSS_INFERENCE_KEY", "not-needed")
        logger.info("Configured column classification endpoint: %s", full_url)


def _validate_flat_tabular_data(data: pd.DataFrame) -> None:
    """Reject non-tabular input with a clear, actionable error.

    Safe Synthesizer expects one scalar value per cell. Nested values (lists or
    dicts, e.g. from a JSON/JSONL fileset with array fields) otherwise crash deep
    inside preflight with an opaque ``TypeError: unhashable type: 'list'``. Naming
    the offending columns here turns that into something the user can act on.
    """
    nested_columns = [
        str(column)
        for column in data.columns
        # Only object-dtype columns can hold Python containers; skip the rest.
        if data[column].dtype == object and data[column].map(lambda value: isinstance(value, (list, dict, set))).any()
    ]
    if nested_columns:
        raise ParameterError(
            "Input data is not flat tabular: column(s) "
            f"{', '.join(nested_columns)} contain nested list/dict values. "
            "Safe Synthesizer expects one scalar value per cell — flatten these "
            "columns or serialize them to strings (e.g. JSON) before running."
        )


def run_config(
    job_config: SafeSynthesizerJobConfig,
    data_source: pd.DataFrame,
    save_path: Path,
    *,
    adapter_location: str | Path | None = None,
) -> tuple[SafeSynthesizerResults, Path | None]:
    """Run NSS against a validated config and already-loaded data source."""
    _validate_flat_tabular_data(data_source)
    enable_synthesis: bool = job_config.enable_synthesis
    logger.info("enable_synthesis=%s", enable_synthesis)
    logger.info("Nemo Safe Synthesizer task job config: %s", job_config.model_dump_json(indent=2))
    logger.info("Using save_path: %s", save_path)

    http_backoff = None
    original_max_retries = None
    try:
        from huggingface_hub.utils._http import http_backoff

        if http_backoff.__kwdefaults__ is not None:
            original_max_retries = http_backoff.__kwdefaults__.get("max_retries")
            http_backoff.__kwdefaults__["max_retries"] = 0

        if not enable_synthesis:
            job_config.config.data.holdout = 0
            job_config.config.data.max_holdout = 0

        if enable_synthesis and is_adapter_reuse_requested(job_config):
            location = adapter_location or job_config.config.training.pretrained_model
            return run_generation_from_prior_adapter(
                job_config,
                data_source,
                save_path,
                adapter_location=str(location) if location is not None else None,
            )

        ss: SafeSynthesizer = SafeSynthesizer(config=job_config.config, save_path=save_path).with_data_source(
            data_source
        )
        if enable_synthesis:
            ss.run()
        else:
            ss.process_data()
            total_time = time.monotonic() - (ss._total_start or time.monotonic())
            pii_replaced_df = getattr(ss, "_training_df", None)
            if pii_replaced_df is None:
                # Older upstream builds used _train_df; keep a fallback so the
                # task does not depend on exactly one NSS internals name.
                pii_replaced_df = getattr(ss, "_train_df", None)
            if pii_replaced_df is None:
                raise RuntimeError("process_data() completed but processed training data is unavailable")
            from nemo_safe_synthesizer.results import make_nss_results

            ss.results = make_nss_results(generate_results=pii_replaced_df, total_time=total_time)
    finally:
        if http_backoff is not None and http_backoff.__kwdefaults__ is not None and original_max_retries is not None:
            http_backoff.__kwdefaults__["max_retries"] = original_max_retries

    adapter_path = ss._workdir.adapter_path if ss._workdir and enable_synthesis else None
    if adapter_path is not None:
        embed_run_config_in_adapter(adapter_path)
    return ss.results, adapter_path


def run_from_env() -> None:
    """Run in the platform task-container environment."""
    initialize_observability()
    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, "default")
    sdk = get_platform_sdk()
    files_url = get_platform_config().get_service_url("files")
    if files_url:
        logger.info("Initializing model weights from Files API...")
        results = init_models_sync(files_api_url=files_url)
        failed = [m for m, success in results.items() if not success]
        if failed:
            logger.warning("Some models failed to download: %s", failed)
        else:
            logger.info("Successfully initialized %d model(s)", len(results))
    else:
        logger.debug("Files API URL not configured, skipping model initialization")

    _setup_classify_endpoint()

    data_source_url = os.environ.get("DATA_SOURCE")
    if data_source_url is None:
        raise ValueError("DATA_SOURCE is not set")

    data_source = download_from_fileset(fileset_url=data_source_url)
    config_file_path = os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR)
    if not config_file_path:
        raise ValueError(f"{NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} is not set")
    with open(config_file_path, "r", encoding="utf-8") as f:
        raw_job_config: dict = json.load(f)

    job_config = SafeSynthesizerJobConfig.model_validate(raw_job_config)
    save_path = Path(os.environ.get(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, DEFAULT_TASK_STORAGE_PATH))
    pretrained_model_tmp, adapter_path = _resolve_pretrained_model(job_config, workspace=workspace, sdk=sdk)
    try:
        result, adapter_path = run_config(
            job_config,
            data_source,
            save_path,
            adapter_location=adapter_path,
        )
        upload_results(result=result, adapter_path=adapter_path)
    finally:
        if pretrained_model_tmp is not None:
            pretrained_model_tmp.cleanup_tmp_dir()


def main(argv: list[str] | None = None) -> None:
    """Run the task entry point from the platform task-container environment."""
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        raise SystemExit(
            "Safe Synthesizer task subcommands are not supported. "
            "Create Safe Synthesizer jobs through the platform Jobs service."
        )
    run_from_env()


if __name__ == "__main__":
    main()
