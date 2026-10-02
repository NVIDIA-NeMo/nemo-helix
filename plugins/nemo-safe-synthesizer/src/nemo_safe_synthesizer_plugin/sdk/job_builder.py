# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder for Safe Synthesizer jobs created through the plugin SDK."""

from __future__ import annotations

import logging
import random
import string
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from nemo_helix import NeMoHelix
from nemo_safe_synthesizer_plugin.sdk.job import SafeSynthesizerJob
from typing_extensions import Self

logger = logging.getLogger(__name__)

_ConfigInput = Any


def _to_dict(config: _ConfigInput) -> dict[str, Any]:
    """Coerce a Pydantic model, dict, or None to a plain dict."""
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if isinstance(config, dict):
        return config.copy()
    raise ValueError(f"Config must be a dict, a Pydantic model, or None; got {type(config)!r}")


def _merge_config(config: _ConfigInput, kwargs: dict[str, Any]) -> dict[str, Any]:
    base = _to_dict(config)
    base.update(kwargs)
    return base


class SafeSynthesizerJobBuilder:
    """Fluent builder for Safe Synthesizer plugin jobs."""

    def __init__(self, client: NeMoHelix, workspace: str = "default"):
        self._client = client
        self._workspace = workspace

        self._hf_token_secret: str | None = None
        self._classify_model_provider: str | None = None
        self._pretrained_model_job: str | None = None
        self._data_source: pd.DataFrame | str | Path | None = None
        self._data_source_path: str | None = None

        self._enable_synthesis = False
        self._enable_replace_pii = False

        self._data_config: dict[str, Any] = {}
        self._training_config: dict[str, Any] = {}
        self._generation_config: dict[str, Any] = {}
        self._evaluation_config: dict[str, Any] = {}
        self._privacy_config: dict[str, Any] | None = None
        self._time_series_config: dict[str, Any] = {}
        self._replace_pii_config: dict[str, Any] = {}

    def with_data_source(self, df_source: pd.DataFrame | str | Path) -> Self:
        """Set the data source as a DataFrame or local data file path."""
        self._data_source = df_source
        self._data_source_path = None
        return self

    def synthesize(self) -> Self:
        """Enable data synthesis for the job run."""
        self._enable_synthesis = True
        return self

    def with_data(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure data parameters."""
        self._data_config = _merge_config(config, kwargs)
        return self

    def with_train(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure training hyperparameters and enable synthesis."""
        self._training_config = _merge_config(config, kwargs)
        self._enable_synthesis = True
        return self

    def with_generate(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure generation parameters and enable synthesis."""
        self._generation_config = _merge_config(config, kwargs)
        self._enable_synthesis = True
        return self

    def with_evaluate(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure evaluation parameters."""
        self._evaluation_config = _merge_config(config, kwargs)
        return self

    def with_differential_privacy(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure differential privacy parameters."""
        self._privacy_config = _merge_config(config, kwargs)
        return self

    def with_time_series(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure time-series parameters."""
        self._time_series_config = _merge_config(config, kwargs)
        return self

    def with_replace_pii(self, config: _ConfigInput = None, **kwargs: Any) -> Self:
        """Configure and enable PII replacement."""
        self._replace_pii_config = _merge_config(config, kwargs)
        self._enable_replace_pii = True
        return self

    def with_classify_model_provider(self, provider_name: str) -> Self:
        """Configure the model provider used by PII replacement column classification.

        The provider is included in the job spec only when PII replacement is enabled with
        ``with_replace_pii()``.
        """
        if "/" in provider_name:
            self._classify_model_provider = provider_name
        else:
            self._classify_model_provider = f"{self._workspace}/{provider_name}"
        logger.info("Configured classify model provider: %s", self._classify_model_provider)
        return self

    def with_hf_token_secret(self, secret_name: str) -> Self:
        """Configure Hugging Face authentication through a platform secret."""
        self._hf_token_secret = secret_name
        return self

    def with_pretrained_model_job(self, job_name: str) -> Self:
        """Reuse a previous Safe Synthesizer job's adapter for generation-only synthesis.

        Args:
            job_name: Completed job name in the current workspace, or a fully-qualified
                ``<workspace>/<job>`` reference.
        """
        self._pretrained_model_job = job_name
        self._enable_synthesis = True
        return self

    def resolve_job_config(self) -> Self:
        """Upload input data and validate the final job configuration without creating a job."""
        self._resolve_datasource()
        self._build_job_spec()
        return self

    def create_job(self, **kwargs: Any) -> SafeSynthesizerJob:
        """Upload input data and create the Safe Synthesizer job."""
        self._resolve_datasource()
        response = self._client.safe_synthesizer.jobs.create(
            workspace=self._workspace,
            spec=self._build_job_spec(),
            **kwargs,
        )
        return SafeSynthesizerJob(response.name, self._client, workspace=self._workspace)

    def _resolve_datasource(self, **kwargs: Any) -> None:
        if self._data_source_path is not None:
            return
        df: pd.DataFrame
        if isinstance(self._data_source, pd.DataFrame):
            df = self._data_source
        elif isinstance(self._data_source, str | Path):
            data_source_path = Path(self._data_source)
            match data_source_path.suffix.lower():
                case ".parquet":
                    df = pd.read_parquet(data_source_path, **kwargs)
                case ".jsonl":
                    df = cast(pd.DataFrame, pd.read_json(data_source_path, lines=True, **kwargs))
                case ".json":
                    df = cast(pd.DataFrame, pd.read_json(data_source_path, **kwargs))
                case _:
                    df = cast(pd.DataFrame, pd.read_csv(data_source_path, **kwargs))
        else:
            raise ValueError("Data source must be a pandas DataFrame or local data file path")

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            df.to_csv(tmp_path, index=False)
            file_name = f"dataset{self._generate_random_string()}.csv"
            self._data_source_path = self._upload_to_fileset(
                dataset_path=tmp_path,
                filename=file_name,
                fileset_name="safe-synthesizer-inputs",
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _build_job_spec(self) -> dict[str, Any]:
        if not self._enable_replace_pii and not self._enable_synthesis:
            raise ValueError("Data synthesis and/or replace PII must be enabled")
        if not self._data_source_path:
            raise ValueError("No data source path found after uploading dataset")

        pii_config: dict[str, Any] | None = None
        if self._enable_replace_pii:
            pii_config = self._replace_pii_config.copy()
            if self._classify_model_provider:
                globals_cfg = pii_config.setdefault("globals", {})
                classify_cfg = globals_cfg.setdefault("classify", {})
                classify_cfg["classify_model_provider"] = self._classify_model_provider

        nss_config: dict[str, Any] = {
            "enable_synthesis": self._enable_synthesis,
            "enable_replace_pii": self._enable_replace_pii,
            "data": self._data_config,
            "training": self._training_config,
            "generation": self._generation_config,
            "evaluation": self._evaluation_config,
            "time_series": self._time_series_config,
        }
        if self._privacy_config is not None:
            nss_config["privacy"] = self._privacy_config
        if pii_config is not None:
            nss_config["replace_pii"] = pii_config

        spec: dict[str, Any] = {
            "data_source": self._data_source_path,
            "config": nss_config,
        }
        if self._hf_token_secret:
            spec["hf_token_secret"] = self._hf_token_secret
        if self._pretrained_model_job:
            spec["pretrained_model_job"] = self._pretrained_model_job
        return spec

    def _upload_to_fileset(self, dataset_path: str | Path, filename: str, fileset_name: str) -> str:
        dataset_path = self._validate_dataset_path(dataset_path)
        self._client.files.upload(
            local_path=str(dataset_path),
            remote_path=filename,
            fileset=fileset_name,
            workspace=self._workspace,
            fileset_auto_create=True,
        )
        return f"{self._workspace}/{fileset_name}#{filename}"

    def _generate_random_string(self, length: int = 6) -> str:
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    @staticmethod
    def _validate_dataset_path(dataset_path: str | Path) -> Path:
        path = Path(dataset_path)
        if not path.is_file():
            raise ValueError("To upload a dataset, you must provide a valid file path.")
        if path.suffix not in {".parquet", ".csv", ".json", ".jsonl"}:
            raise ValueError("Dataset files must be in parquet, csv, json, or jsonl format.")
        return path
