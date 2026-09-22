# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from nemo_insights_plugin import preflight
from nemo_insights_plugin.contracts.checks import format_report, required_failures
from nemo_insights_plugin.preflight import (
    AnalysisProbes,
    check_environment,
    check_ethos,
    check_profile,
)
from nemo_insights_plugin.profile import AnalysisProfile
from nemo_helix import NeMoHelixError


async def always_queryable(base_url: str, workspace: str, agent: str) -> bool:
    return True


async def never_queryable(base_url: str, workspace: str, agent: str) -> bool:
    return False


@pytest.fixture(autouse=True)
def configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight,
        "configured_model_refs",
        lambda: SimpleNamespace(default="default/gpt-5", fast="default/gpt-5-mini"),
    )


def test_missing_model_pair_is_required_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_models() -> object:
        raise ValueError("No default model is configured")

    monkeypatch.setattr(preflight, "configured_model_refs", missing_models)
    results = asyncio.run(
        check_environment(
            agent="a",
            workspace="default",
            base_url="http://localhost:8080",
            profile_dir=tmp_path,
            probes=AnalysisProbes(
                http_ok=lambda base_url: True,
                workspace_ok=always_queryable,
            ),
        )
    )

    assert any(result.name == "agent-models" and result.status == "fail" for result in results)
    assert required_failures(results)


def test_check_environment_without_profile_skips_workspace_probe(tmp_path: Path) -> None:
    results = asyncio.run(
        check_environment(
            agent=None,
            workspace=None,
            base_url="http://localhost:8080",
            profile_dir=None,
            probes=AnalysisProbes(
                http_ok=lambda base_url: True,
                workspace_ok=never_queryable,
            ),
        )
    )

    assert [result.name for result in results] == ["agent-models", "platform-reachable"]


def test_default_http_probe_treats_invalid_base_url_as_unreachable() -> None:
    assert preflight._default_http_ok("http://localhost:not-a-port") is False


def test_workspace_query_failure_is_advisory(tmp_path: Path) -> None:
    results = asyncio.run(
        check_environment(
            agent="a",
            workspace="missing",
            base_url="http://localhost:8080",
            profile_dir=tmp_path,
            probes=AnalysisProbes(
                http_ok=lambda base_url: True,
                workspace_ok=never_queryable,
            ),
        )
    )

    warning = next(result for result in results if result.name == "workspace-query")
    assert warning.status == "warn"
    assert warning.severity == "advisory"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("OIDC discovery failed", request=httpx.Request("GET", "https://platform.example")),
        NeMoHelixError("SDK initialization failed"),
        RuntimeError("NeMoHelix client initialization failed: invalid context"),
        ValueError("invalid remote configuration"),
        OSError("could not read SDK configuration"),
    ],
)
def test_remote_workspace_probe_treats_client_construction_failures_as_advisory(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail_to_construct(base_url: str) -> object:
        raise error

    monkeypatch.setattr(preflight, "make_client", fail_to_construct)

    assert asyncio.run(preflight._default_workspace_ok("https://platform.example", "default", "agent")) is False


def test_profile_and_ethos_failures_are_required(tmp_path: Path) -> None:
    profile_results = check_profile(None, None)
    ethos_results = check_ethos(None, "configured Ethos does not exist")

    assert required_failures(profile_results) == profile_results
    assert required_failures(ethos_results) == ethos_results


def test_ethos_invalid_utf8_is_required_failure(tmp_path: Path) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_bytes(b"\xff\xfe")

    results = check_ethos(ethos, None)

    assert results[0].status == "fail"
    assert results[0].severity == "required"
    assert "UTF-8" in results[0].message
    assert results[0].hint == "ensure the file is readable and encoded as UTF-8"


def test_ethos_unreadable_is_required_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Agent", encoding="utf-8")
    original_read_text = Path.read_text

    def deny_ethos_read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == ethos:
            raise PermissionError("permission denied")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_ethos_read)

    results = check_ethos(ethos, None)

    assert results[0].status == "fail"
    assert results[0].severity == "required"
    assert "permission denied" in results[0].message
    assert results[0].hint == "ensure the file is readable and encoded as UTF-8"


def test_healthy_setup_formats_grouped_report(tmp_path: Path) -> None:
    profile = AnalysisProfile(agent="a", profile_dir=tmp_path)
    (tmp_path / "ETHOS.md").write_text("# Agent", encoding="utf-8")
    results = check_profile(profile, None) + check_ethos(tmp_path / "ETHOS.md", None)
    results += asyncio.run(
        check_environment(
            agent="a",
            workspace="default",
            base_url="http://localhost:8080",
            profile_dir=tmp_path,
            probes=AnalysisProbes(
                http_ok=lambda base_url: True,
                workspace_ok=always_queryable,
            ),
        )
    )

    report = format_report(results)

    assert "Profile\n  ✓ profile for agent 'a'" in report
    assert "Models\n  ✓ default=default/gpt-5; fast=default/gpt-5-mini" in report
    assert "Platform\n  ✓ http://localhost:8080 reachable" in report
    assert not required_failures(results)
