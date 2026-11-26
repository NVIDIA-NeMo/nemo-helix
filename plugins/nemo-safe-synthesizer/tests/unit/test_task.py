# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.mark.parametrize("disabled_checks", [None, ["token_budget", "dataset.size"]])
def test_run_config_pii_only_disables_dataset_size_floor(tmp_path, monkeypatch, disabled_checks):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)

    small_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [1, 2]})
    observed_disabled_checks = []

    class FakeSafeSynthesizer:
        def __init__(self, config, save_path):
            # Exercise the real sparse-config rebuild used by AutoConfigResolver.
            self.config = config.with_config_patch({})
            self._total_start = None
            self._training_df = None
            self._workdir = None

        def with_data_source(self, data_source):
            return self

        def process_data(self):
            assert self.config.data.holdout == 0
            assert self.config.data.max_holdout == 0
            observed_disabled_checks.extend(self.config.preflight.disabled_checks)
            self._training_df = small_df

    monkeypatch.setattr(task_main, "SafeSynthesizer", FakeSafeSynthesizer)
    results_module = ModuleType("nemo_safe_synthesizer.results")
    setattr(
        results_module,
        "make_nss_results",
        lambda generate_results, total_time=None: SimpleNamespace(synthetic_data=generate_results),
    )
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.results", results_module)
    job_config = task_main.SafeSynthesizerJobConfig.model_validate(
        {
            "data_source": "default/data#input.csv",
            "config": {"enable_synthesis": False, "enable_replace_pii": False},
        }
    )

    assert "data" not in job_config.config.model_fields_set
    assert "preflight" not in job_config.config.model_fields_set
    if disabled_checks is not None:
        job_config.config = job_config.config.with_config_patch({"preflight": {"disabled_checks": disabled_checks}})
    original_config = job_config.config.model_dump()

    task_main.run_config(job_config, small_df, tmp_path)

    assert observed_disabled_checks == (disabled_checks if disabled_checks is not None else ["dataset.size"])
    assert job_config.config.model_dump() == original_config
    assert len(small_df) < 200


def test_run_from_env_reports_missing_config_path(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    monkeypatch.setattr(task_main, "initialize_observability", lambda: None)
    monkeypatch.setattr(task_main, "init_models_sync", lambda: {})
    monkeypatch.setattr(task_main, "download_from_fileset", lambda fileset_url: pd.DataFrame({"value": [1]}))
    monkeypatch.setenv("DATA_SOURCE", "default/data#input.csv")
    monkeypatch.delenv(task_main.NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR, raising=False)

    with pytest.raises(ValueError, match=f"{task_main.NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} is not set"):
        task_main.run_from_env()


def test_resolve_pretrained_model_uses_current_attempt_selector(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    job_config = task_main.SafeSynthesizerJobConfig.model_validate(
        {
            "data_source": "default/data#input.csv",
            "pretrained_model_job": "prior-safe-synth-job",
            "config": {},
        }
    )
    client = MagicMock()
    jobs = MagicMock()
    result_response = MagicMock()
    result_response.data.return_value = SimpleNamespace(
        artifact_url="default/job-results-prior#results/attempt-1/adapter"
    )
    jobs.get_job_result.return_value = result_response
    monkeypatch.setattr(task_main.JobsClient, "from_client", classmethod(lambda _cls, _client: jobs))

    files = MagicMock()
    monkeypatch.setattr(task_main.FilesClient, "from_client", classmethod(lambda _cls, _client: files))
    downloaded = SimpleNamespace(path=tmp_path / "adapter")
    file_manager = MagicMock()
    file_manager.download_from_url.return_value = downloaded
    file_manager_cls = MagicMock(return_value=file_manager)
    monkeypatch.setattr(task_main, "FilesetFileManager", file_manager_cls)

    tmp_dir_path, adapter_path = task_main._resolve_pretrained_model(
        job_config,
        workspace="default",
        client=client,
    )

    jobs.get_job_attempt.assert_not_called()
    jobs.get_job_result.assert_called_once_with(
        name="adapter",
        job="prior-safe-synth-job",
        workspace="default",
    )
    assert tmp_dir_path is downloaded
    assert adapter_path == downloaded.path


PROVIDER_PATH = "/apis/inference-gateway/v2/workspaces/default/provider/my-nim/-/v1"


@pytest.fixture
def pii_task(tmp_path, monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    state = SimpleNamespace(proxy_open=False, proxy_entered=0, proxy_origin="http://127.0.0.1:4321")
    calls = {"with_inference": [], "proxy_open_during_run": []}

    @contextlib.contextmanager
    def fake_platform_auth_proxy():
        state.proxy_entered += 1
        state.proxy_open = True
        try:
            yield state.proxy_origin
        finally:
            state.proxy_open = False

    class FakeSafeSynthesizer:
        def __init__(self, config, save_path):
            self._total_start = None
            self._training_df = None
            self._workdir = None

        def with_data_source(self, data_source):
            return self

        def with_inference(self, endpoint_url, model_id, *, timeout_seconds=None):
            calls["with_inference"].append((endpoint_url, model_id, timeout_seconds))
            return self

        def process_data(self):
            calls["proxy_open_during_run"].append(state.proxy_open)
            self._training_df = pd.DataFrame({"name": ["REDACTED"]})

    results_module = ModuleType("nemo_safe_synthesizer.results")
    setattr(results_module, "make_nss_results", lambda generate_results, total_time=None: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo_safe_synthesizer.results", results_module)
    monkeypatch.setattr(task_main, "SafeSynthesizer", FakeSafeSynthesizer)
    monkeypatch.setattr(task_main, "platform_auth_proxy", fake_platform_auth_proxy)
    monkeypatch.setattr(task_main, "_pii_model_id", lambda job_config: "nvidia/pii-model")
    monkeypatch.setattr(task_main, "initialize_observability", lambda: None)
    monkeypatch.setattr(task_main, "init_models_sync", lambda: {})
    monkeypatch.setattr(task_main, "get_nemo_client", MagicMock)
    monkeypatch.setattr(task_main, "download_from_fileset", lambda fileset_url: pd.DataFrame({"name": ["Alice"]}))
    monkeypatch.setattr(task_main, "upload_results", lambda result, adapter_path=None: None)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_source": "default/data#input.csv",
                "config": {"enable_synthesis": False, "enable_replace_pii": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_SOURCE", "default/data#input.csv")
    monkeypatch.setenv(task_main.NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR, str(config_path))
    monkeypatch.setenv(task_main.EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, str(tmp_path / "save"))
    monkeypatch.setenv("CLASSIFY_LLM_ENDPOINT_PATH", PROVIDER_PATH)
    monkeypatch.setenv("NHX_MODELS_URL", "http://models.test/")
    for name in ("NSS_INFERENCE_ENDPOINT", "NSS_INFERENCE_KEY", "NSS_INFERENCE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    from nemo_safe_synthesizer_plugin.tasks.safe_synthesizer.pii_inference import PII_INFERENCE_TIMEOUT_SECONDS

    return SimpleNamespace(main=task_main, state=state, calls=calls, timeout_seconds=PII_INFERENCE_TIMEOUT_SECONDS)


def test_pii_inference_goes_through_the_auth_proxy_for_the_whole_run(pii_task):
    pii_task.main.run_from_env()

    assert pii_task.calls["with_inference"] == [
        (
            f"http://127.0.0.1:4321{PROVIDER_PATH}",
            "nvidia/pii-model",
            pii_task.timeout_seconds,
        )
    ]
    assert pii_task.timeout_seconds == 300
    assert pii_task.calls["proxy_open_during_run"] == [True]
    assert pii_task.state.proxy_open is False


def test_pii_inference_falls_back_to_the_direct_platform_url_without_a_proxy(pii_task):
    pii_task.state.proxy_origin = None

    pii_task.main.run_from_env()

    assert pii_task.calls["with_inference"] == [
        (f"http://models.test{PROVIDER_PATH}", "nvidia/pii-model", pii_task.timeout_seconds)
    ]


def test_without_a_model_provider_no_proxy_starts_and_inference_is_left_to_nss(pii_task, monkeypatch):
    monkeypatch.delenv("CLASSIFY_LLM_ENDPOINT_PATH")

    pii_task.main.run_from_env()

    assert pii_task.state.proxy_entered == 0
    assert pii_task.calls["with_inference"] == []
    assert pii_task.calls["proxy_open_during_run"] == [False]


def test_a_provider_without_a_model_id_starts_no_proxy(pii_task, monkeypatch):
    monkeypatch.setattr(pii_task.main, "_pii_model_id", lambda job_config: None)

    pii_task.main.run_from_env()

    assert pii_task.state.proxy_entered == 0
    assert pii_task.calls["with_inference"] == []


def test_pii_inference_no_longer_configures_nss_through_the_environment(pii_task):
    pii_task.main.run_from_env()

    assert not any(name.startswith("NSS_INFERENCE_") for name in pii_task.main.os.environ)


def test_pii_model_id_reads_the_replace_pii_llm_model(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)
    from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerParameters

    llm = SimpleNamespace(model_id="nvidia/pii-model")
    with_pii = task_main.SafeSynthesizerJobConfig.model_construct(
        config=SafeSynthesizerParameters.model_construct(replace_pii=SimpleNamespace(llm=llm))
    )
    without_pii = task_main.SafeSynthesizerJobConfig.model_construct(
        config=SafeSynthesizerParameters.model_construct(replace_pii=None)
    )

    assert task_main._pii_model_id(with_pii) == "nvidia/pii-model"
    assert task_main._pii_model_id(without_pii) is None


def test_main_rejects_task_subcommands(monkeypatch):
    task_main = import_task_main_without_heavy_runtime(monkeypatch)

    with pytest.raises(SystemExit, match="platform Jobs service"):
        task_main.main(["run-" + "local"])
