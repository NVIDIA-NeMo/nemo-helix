# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PLUGIN_ROOT.parents[1]


def test_platform_stack_dependencies_resolve_through_the_workspace() -> None:
    """The plugin tracks the monorepo it lives in, never a pinned copy of it."""
    project = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    workspace = tomllib.loads((WORKSPACE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = workspace["tool"]["uv"]["sources"]

    for package_name in ("nemo-platform", "nemo-platform-plugin", "nemo-insights-plugin"):
        assert project["project"]["dependencies"].count(package_name) == 1
        assert sources[package_name] == {"workspace": True}

    assert "uv" not in project["tool"]
    assert str(PLUGIN_ROOT.relative_to(WORKSPACE_ROOT)) in workspace["tool"]["uv"]["workspace"]["members"]


def test_runtime_dependencies_match_retained_imports() -> None:
    project = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])

    assert "pydantic>=2" in dependencies
    assert "opentelemetry-proto>=1.42.1" in dependencies
    assert "protobuf>=6.0.0" in dependencies
    assert "nemo-insights-plugin" in dependencies
    assert all("opentelemetry-sdk" not in dependency for dependency in dependencies)
    assert all("opentelemetry-exporter-otlp" not in dependency for dependency in dependencies)
    assert all("pydantic-ai" not in dependency for dependency in dependencies)
    assert all("tzdata" not in dependency for dependency in dependencies)


def test_shared_contract_modules_are_importable(tmp_path: Path) -> None:
    from nemo_insights_plugin.contracts.checks import CheckResult, format_report
    from nemo_insights_plugin.contracts.insights import InsightsFileError, load_insights_document
    from nemo_insights_plugin.contracts.profile import DEFAULT_BASE_URL, discover_profile, resolve_base_url

    result = CheckResult(
        name="profile",
        group="profile",
        status="pass",
        severity="required",
        message="ready",
    )
    insight = tmp_path / "insight.yaml"
    insight.write_text("id: one\n", encoding="utf-8")

    assert format_report([result]) == "Profile\n  ✓ ready"
    assert issubclass(InsightsFileError, ValueError)
    assert load_insights_document(insight) == {"id": "one"}
    assert discover_profile(tmp_path) is None
    assert resolve_base_url(None, {}) == DEFAULT_BASE_URL
