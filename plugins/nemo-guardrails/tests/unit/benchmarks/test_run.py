# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import yaml
from nemo_guardrails_plugin.benchmarks.constants import VARIANT_WITH_GUARDRAILS
from nemo_guardrails_plugin.benchmarks.paths import RunPaths, build_run_paths
from nemo_guardrails_plugin.benchmarks.run import _build_nhx_process, _run_benchmark, _smoke_test, parse_args
from nemo_guardrails_plugin.benchmarks.seeding import SeededResources
from nemo_helix_plugin.client.client import NemoClient


def _run_paths(tmp_path: Path) -> RunPaths:
    return build_run_paths(
        nhx_repo_root=tmp_path / "nemo-helix",
        nemoguardrails_repo_root=tmp_path / "NeMo-Guardrails",
        run_id="test-run",
    )


def _seeded_resources() -> SeededResources:
    return SeededResources(
        workspace="benchmark",
        app_provider_name="app-provider",
        cs_provider_name="cs-provider",
        app_model_entity="benchmark/app-model",
        cs_model_entity="benchmark/cs-model",
        guardrail_config_name="content-safety",
        vm_name="guardrails-vm",
        no_guardrails_vm_name="control-vm",
    )


def test_nhx_process_uses_benchmark_service_subset(tmp_path: Path) -> None:
    process = _build_nhx_process(_run_paths(tmp_path))

    assert process.cmd == [
        sys.executable,
        "-m",
        "nemo_guardrails_plugin.benchmarks.platform_runner",
    ]


def test_smoke_test_retries_until_virtual_model_route_is_ready(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(404, request=request, json={"error": "not ready"})
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("nemo_guardrails_plugin.benchmarks.run.sleep", lambda _seconds: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = NemoClient(base_url="http://platform.test", http_client=http_client)

        _smoke_test(client, _seeded_resources())

    assert attempts == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1,2", (1, 2)), ("32,16", (16, 32)), ("4, 8,4", (4, 8)), ("8", (8,))],
)
def test_concurrency_flag_parses_levels(value: str, expected: tuple[int, ...]) -> None:
    assert parse_args(["--concurrency", value]).concurrency == expected


def test_concurrency_flag_defaults_to_every_configured_level() -> None:
    assert parse_args([]).concurrency is None


@pytest.mark.parametrize("value", ["", "a,b", "1,x", "0", "-1,2"])
def test_concurrency_flag_rejects_invalid_levels(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--concurrency", value])


def test_run_benchmark_sweeps_only_selected_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _run_paths(tmp_path)
    paths.config_template.parent.mkdir(parents=True)
    paths.config_template.write_text(
        yaml.safe_dump(
            {
                "batch_name": "sweep",
                "base_config": {"model": "placeholder", "benchmark_duration": 60},
                "sweeps": {"concurrency": [1, 2, 4, 8, 16, 32]},
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_aiperf_sweep(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("nemo_guardrails_plugin.benchmarks.run.run_aiperf_sweep", fake_run_aiperf_sweep)

    outcome = _run_benchmark(
        variant=VARIANT_WITH_GUARDRAILS,
        paths=paths,
        seeded=_seeded_resources(),
        aiperf_python=Path(sys.executable),
        concurrencies=(4, 8),
    )

    [call] = calls
    runtime_config = call["runtime_config"]
    assert isinstance(runtime_config, Path)
    written = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert written["sweeps"]["concurrency"] == [4, 8]
    assert written["base_config"] == {"model": "benchmark/guardrails-vm", "benchmark_duration": 60}
    # Results stay under the variant so shards merge; file names carry the
    # levels so shards of one variant don't overwrite each other's.
    assert outcome.output_dir == paths.aiperf_output_dir / VARIANT_WITH_GUARDRAILS
    assert runtime_config.name == "aiperf_config_with-guardrails_c4-8.yaml"
    assert call["log_path"] == paths.log_dir / "aiperf_with-guardrails_c4-8.log"
