# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def import_task_main_without_heavy_runtime(monkeypatch):
    pytest.importorskip("nemo_safe_synthesizer.config.job")
    library_builder = ModuleType("nemo_safe_synthesizer.sdk.library_builder")
    setattr(library_builder, "SafeSynthesizer", object)
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.sdk.library_builder", library_builder)
    return importlib.import_module("nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.__main__")


def test_validate_flat_tabular_data_rejects_nested_columns(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "tools": [[{"name": "f"}], []],
            "metadata": [{"a": 1}, {"a": 2}],
        }
    )

    with pytest.raises(task_main.ParameterError, match="tools, metadata"):
        task_main._validate_flat_tabular_data(data)


def test_validate_flat_tabular_data_allows_flat_columns(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    data = pd.DataFrame({"name": ["a", "b"], "age": [1, 2], "score": [0.5, 0.9]})

    task_main._validate_flat_tabular_data(data)


def test_run_config_pii_only_uses_processed_training_df(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    processed = pd.DataFrame({"name": ["REDACTED"], "age": [1]})

    class FakeSafeSynthesizer:
        def __init__(self, config, save_path):
            self._total_start = None
            self._training_df = None
            self._workdir = None

        def with_data_source(self, data_source):
            return self

        def process_data(self):
            self._training_df = processed

    monkeypatch.setattr(task_main, "SafeSynthesizer", FakeSafeSynthesizer)
    results_module = ModuleType("nemo_safe_synthesizer.results")
    setattr(
        results_module,
        "make_nss_results",
        lambda generate_results, total_time=None: SimpleNamespace(
            synthetic_data=generate_results,
            summary=SimpleNamespace(model_dump=lambda: {"row_count": len(generate_results)}),
            evaluation_report_html=None,
        ),
    )
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.results", results_module)
    job_config = task_main.SafeSynthesizerJobConfig.model_validate(
        {
            "data_source": "default/data#input.csv",
            "config": {
                "enable_synthesis": False,
                "enable_replace_pii": False,
            },
        }
    )

    result, adapter_path = task_main.run_config(
        job_config,
        pd.DataFrame({"name": ["Alice"], "age": [1]}),
        tmp_path,
    )

    assert adapter_path is None
    pd.testing.assert_frame_equal(result.synthetic_data, processed)


def test_run_from_env_reports_missing_config_path(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    monkeypatch.setattr(task_main, "initialize_observability", lambda: None)
    monkeypatch.setattr(task_main, "get_platform_config", lambda: SimpleNamespace(get_service_url=lambda _name: None))
    monkeypatch.setattr(task_main, "_setup_classify_endpoint", lambda: None)
    monkeypatch.setattr(task_main, "download_from_fileset", lambda fileset_url: pd.DataFrame({"value": [1]}))
    monkeypatch.setenv("DATA_SOURCE", "default/data#input.csv")
    monkeypatch.delenv(task_main.NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR, raising=False)

    with pytest.raises(ValueError, match=f"{task_main.NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} is not set"):
        task_main.run_from_env()


def test_setup_classify_endpoint_sets_upstream_safe_synthesizer_env(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    monkeypatch.setenv(
        "CLASSIFY_LLM_ENDPOINT_PATH", "/apis/inference-gateway/v2/workspaces/default/provider/my-nim/-/v1"
    )
    monkeypatch.setenv("NHX_MODELS_URL", "http://models.test")
    monkeypatch.delenv("NSS_INFERENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("NSS_INFERENCE_KEY", raising=False)

    task_main._setup_classify_endpoint()

    assert (
        task_main.os.environ["NSS_INFERENCE_ENDPOINT"]
        == "http://models.test/apis/inference-gateway/v2/workspaces/default/provider/my-nim/-/v1"
    )
    assert task_main.os.environ["NSS_INFERENCE_KEY"] == "not-needed"


def test_setup_classify_endpoint_preserves_existing_inference_key(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    monkeypatch.setenv("CLASSIFY_LLM_ENDPOINT_PATH", "/route")
    monkeypatch.setenv("NHX_MODELS_URL", "http://models.test/")
    monkeypatch.setenv("NSS_INFERENCE_KEY", "real-key")

    task_main._setup_classify_endpoint()

    assert task_main.os.environ["NSS_INFERENCE_ENDPOINT"] == "http://models.test/route"
    assert task_main.os.environ["NSS_INFERENCE_KEY"] == "real-key"


def test_main_rejects_task_subcommands(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)

    with pytest.raises(SystemExit, match="platform Jobs service"):
        task_main.main(["run-" + "local"])
