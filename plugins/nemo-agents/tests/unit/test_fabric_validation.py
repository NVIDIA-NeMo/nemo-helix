# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Fabric plan and preflight validation helpers."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import nemo_agents_plugin.fabric.validation as validation
import pytest
from nemo_agents_plugin.agent_config import AgentConfig, load_agent_config
from nemo_agents_plugin.fabric.validation import (
    FabricPreflightError,
    FabricValidationError,
    doctor_fabric_config,
    plan_fabric_config,
    validate_fabric_config,
    validate_platform_agent_config,
)


class _FakeFabricConfigError(Exception):
    pass


class _FakeDoctorReport:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def to_mapping(self) -> dict[str, Any]:
        return self._mapping


class _FakeFabric:
    def __init__(
        self,
        *,
        plan: Any = "plan",
        doctor_report: Any | None = None,
        plan_error: Exception | None = None,
        doctor_error: Exception | None = None,
        doctor_delay: float = 0.0,
    ) -> None:
        self.plan_result = plan
        self.doctor_report = (
            doctor_report if doctor_report is not None else _FakeDoctorReport({"status": "pass", "checks": []})
        )
        self.plan_error = plan_error
        self.doctor_error = doctor_error
        self.doctor_delay = doctor_delay
        self.plan_calls: list[dict[str, Any]] = []
        self.plan_thread_ids: list[int] = []
        self.doctor_calls: list[dict[str, Any]] = []

    def plan(self, fabric_config: Any, *, base_dir: Path | str) -> Any:
        self.plan_calls.append({"fabric_config": fabric_config, "base_dir": base_dir})
        self.plan_thread_ids.append(threading.get_ident())
        if self.plan_error is not None:
            raise self.plan_error
        return self.plan_result

    async def doctor(self, fabric_config: Any, *, base_dir: Path | str) -> Any:
        self.doctor_calls.append({"fabric_config": fabric_config, "base_dir": base_dir})
        if self.doctor_delay:
            await asyncio.sleep(self.doctor_delay)
        if self.doctor_error is not None:
            raise self.doctor_error
        return self.doctor_report


@pytest.fixture()
def fake_fabric_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "Fabric", _FakeFabric)
    monkeypatch.setattr(validation, "FabricConfigError", _FakeFabricConfigError)


@pytest.fixture()
def fake_fabric_stack(fake_fabric_client: None) -> None:
    pass


def _example_platform_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "example-agent",
        "description": "Example Agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
                "model": {
                    "provider": "nvidia",
                    "model": "nvidia/nemotron-3-nano-30b-a3b",
                },
            },
            "codex": {
                "kind": "codex",
                "settings": {
                    "sandbox": "workspace-write",
                },
            },
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": "openai/gpt-5.4",
            },
        },
        "telemetry": {
            "enabled": False,
        },
    }


@pytest.mark.asyncio
class TestValidateFabricConfig:
    async def test_plan_only_does_not_run_doctor(self, fake_fabric_client: None) -> None:
        fabric_config = object()
        fabric = _FakeFabric(plan={"plan": "ok"})

        result = await plan_fabric_config(fabric_config, base_dir=Path("/tmp/agent"), fabric=fabric)

        assert result == {"plan": "ok"}
        assert fabric.plan_calls == [{"fabric_config": fabric_config, "base_dir": Path("/tmp/agent")}]
        assert fabric.doctor_calls == []

    async def test_doctor_only_does_not_run_plan(self, fake_fabric_client: None) -> None:
        fabric_config = object()
        doctor_report = _FakeDoctorReport({"status": "pass", "checks": []})
        fabric = _FakeFabric(doctor_report=doctor_report)

        result = await doctor_fabric_config(fabric_config, base_dir=Path("/tmp/agent"), fabric=fabric)

        assert result is doctor_report
        assert fabric.plan_calls == []
        assert fabric.doctor_calls == [{"fabric_config": fabric_config, "base_dir": Path("/tmp/agent")}]

    async def test_returns_plan_and_doctor_report(self, fake_fabric_client: None) -> None:
        fabric_config = object()
        doctor_report = _FakeDoctorReport({"status": "pass", "checks": [{"name": "adapter", "status": "pass"}]})
        fabric = _FakeFabric(plan={"plan": "ok"}, doctor_report=doctor_report)

        result = await validate_fabric_config(fabric_config, base_dir=Path("/tmp/agent"), fabric=fabric)

        assert result.plan == {"plan": "ok"}
        assert result.doctor_report is doctor_report
        assert fabric.plan_calls == [{"fabric_config": fabric_config, "base_dir": Path("/tmp/agent")}]
        assert fabric.doctor_calls == [{"fabric_config": fabric_config, "base_dir": Path("/tmp/agent")}]

    async def test_runs_plan_off_event_loop_thread(self, fake_fabric_client: None) -> None:
        main_thread_id = threading.get_ident()
        fabric = _FakeFabric()

        await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

        assert fabric.plan_thread_ids
        assert fabric.plan_thread_ids[0] != main_thread_id

    async def test_wraps_plan_errors(self, fake_fabric_client: None) -> None:
        fabric = _FakeFabric(plan_error=_FakeFabricConfigError("bad config"))

        with pytest.raises(FabricValidationError, match="Fabric plan failed: bad config"):
            await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

    async def test_wraps_doctor_errors(self, fake_fabric_client: None) -> None:
        fabric = _FakeFabric(doctor_error=RuntimeError("doctor exploded"))

        with pytest.raises(FabricValidationError, match="Fabric doctor failed: doctor exploded"):
            await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

    async def test_wraps_doctor_timeout(self, fake_fabric_client: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(validation, "FABRIC_VALIDATION_TIMEOUT_SECONDS", 0.01)
        fabric = _FakeFabric(doctor_delay=1.0)

        with pytest.raises(FabricValidationError, match="Fabric doctor timed out after 0.01s"):
            await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

    async def test_preflight_failure_reports_failed_checks(self, fake_fabric_client: None) -> None:
        doctor_report = _FakeDoctorReport(
            {
                "status": "fail",
                "checks": [
                    {"name": "adapter_descriptor", "status": "pass", "message": "ok"},
                    {"name": "requirement.binary", "status": "fail", "message": "binary `codex` missing"},
                    {"name": "environment", "status": "warn", "message": "workspace missing"},
                ],
            }
        )
        fabric = _FakeFabric(doctor_report=doctor_report)

        with pytest.raises(FabricPreflightError) as error_info:
            await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

        assert error_info.value.status == "fail"
        assert error_info.value.failed_checks == [
            "requirement.binary: fail - binary `codex` missing",
            "environment: warn - workspace missing",
        ]

    async def test_preflight_failure_without_checks_reports_fallback(self, fake_fabric_client: None) -> None:
        doctor_report = _FakeDoctorReport({"status": "fail", "checks": []})
        fabric = _FakeFabric(doctor_report=doctor_report)

        with pytest.raises(FabricPreflightError, match="No failing subsection was reported"):
            await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

    async def test_dict_doctor_report_is_supported(self, fake_fabric_client: None) -> None:
        fabric = _FakeFabric(doctor_report={"status": "pass", "checks": []})

        result = await validate_fabric_config(object(), base_dir=Path("/tmp/agent"), fabric=fabric)

        assert result.doctor_report == {"status": "pass", "checks": []}


@pytest.mark.asyncio
class TestValidateHelixAgentConfig:
    async def test_validates_platform_config_dict(self, fake_fabric_stack: None) -> None:
        doctor_report = _FakeDoctorReport({"status": "pass", "checks": [{"name": "adapter", "status": "pass"}]})
        fabric = _FakeFabric(plan={"plan": "ok"}, doctor_report=doctor_report)

        result = await validate_platform_agent_config(
            _example_platform_config(),
            base_dir=Path("/tmp/agent"),
            fabric=fabric,
        )

        assert result.agent_config.name == "example-agent"
        assert result.fabric_config.metadata.name == "example-agent"
        assert result.fabric_config.harness.adapter_id == "nvidia.fabric.hermes"
        assert result.fabric_validation_result.plan == {"plan": "ok"}
        assert result.fabric_validation_result.doctor_report is doctor_report
        assert fabric.plan_calls == [{"fabric_config": result.fabric_config, "base_dir": Path("/tmp/agent")}]
        assert fabric.doctor_calls == [{"fabric_config": result.fabric_config, "base_dir": Path("/tmp/agent")}]

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="NOOA adapter requires Python < 3.14")
    async def test_nooa_coding_example_accepts_real_fabric_preflight(self, tmp_path: Path) -> None:
        example_path = Path(__file__).parents[2] / "examples/nemo-agent-config/agent-nooa-coding.yaml"
        config = load_agent_config(example_path)

        result = await validate_platform_agent_config(config, base_dir=tmp_path)

        report = result.fabric_validation_result.doctor_report.to_mapping()
        assert all(check["status"] == "pass" for check in report["checks"] if check["name"] != "resolution")
        resolution = next(check for check in report["checks"] if check["name"] == "resolution")
        assert resolution["status"] in {"pass", "warn"}
        if resolution["status"] == "warn":
            assert resolution["message"] == "no resolution strategy selected"

    async def test_workflow_missing_resolution_is_accepted(self, tmp_path: Path) -> None:
        config = _example_platform_config()
        del config["default_harness"]
        del config["harnesses"]
        config["workflow"] = {"target_id": "com.example.workflow"}
        report = {
            "status": "warn",
            "checks": [{"name": "resolution", "status": "warn", "message": "no resolution strategy selected"}],
        }

        result = await validate_platform_agent_config(
            config, base_dir=tmp_path, fabric=_FakeFabric(doctor_report=report)
        )

        # Keep the original diagnostics available to callers rather than rewriting
        # the report to look like Fabric returned a passing status.
        assert result.fabric_validation_result.doctor_report is report

    @pytest.mark.parametrize(
        ("status", "extra_checks"),
        [
            ("warn", [{"name": "environment", "status": "warn", "message": "workspace missing"}]),
            ("fail", [{"name": "adapter_target_descriptor", "status": "fail", "message": "target missing"}]),
            ("warn", [{"name": "resolution", "status": "warn", "message": "unsupported strategy"}]),
            ("fail", []),
        ],
    )
    async def test_workflow_other_preflight_problems_remain_fatal(
        self, tmp_path: Path, status: str, extra_checks: list[dict[str, str]]
    ) -> None:
        config = _example_platform_config()
        del config["default_harness"]
        del config["harnesses"]
        config["workflow"] = {"target_id": "nvidia.nooa.coding-agent"}
        report = {
            "status": status,
            "checks": [
                {"name": "resolution", "status": "warn", "message": "no resolution strategy selected"},
                *extra_checks,
            ],
        }

        with pytest.raises(FabricPreflightError) as error_info:
            await validate_platform_agent_config(config, base_dir=tmp_path, fabric=_FakeFabric(doctor_report=report))

        assert error_info.value.status == status
        assert all("no resolution strategy selected" not in check for check in error_info.value.failed_checks)

    async def test_harness_missing_resolution_remains_fatal(self, tmp_path: Path) -> None:
        report = {
            "status": "warn",
            "checks": [{"name": "resolution", "status": "warn", "message": "no resolution strategy selected"}],
        }

        with pytest.raises(FabricPreflightError, match="no resolution strategy selected"):
            await validate_platform_agent_config(
                _example_platform_config(), base_dir=tmp_path, fabric=_FakeFabric(doctor_report=report)
            )

    async def test_validates_agent_config_object(self, fake_fabric_stack: None) -> None:
        agent_config = AgentConfig.model_validate(_example_platform_config())

        result = await validate_platform_agent_config(agent_config, base_dir=Path("/tmp/agent"), fabric=_FakeFabric())

        assert result.agent_config is agent_config

    async def test_selected_harness_is_translated(self, fake_fabric_stack: None) -> None:
        result = await validate_platform_agent_config(
            _example_platform_config(),
            base_dir=Path("/tmp/agent"),
            harness_name="codex",
            fabric=_FakeFabric(),
        )

        assert result.fabric_config.harness.adapter_id == "nvidia.fabric.codex"
        assert result.fabric_config.models["default"].model == "openai/gpt-5.4"

    async def test_invalid_platform_config_is_reported(self, fake_fabric_stack: None) -> None:
        payload = _example_platform_config()
        del payload["default_harness"]

        with pytest.raises(FabricValidationError, match="Invalid Platform agent config"):
            await validate_platform_agent_config(payload, base_dir=Path("/tmp/agent"), fabric=_FakeFabric())

    async def test_translation_error_is_reported(self, fake_fabric_stack: None) -> None:
        payload = _example_platform_config()
        payload["default_harness"] = "custom"
        payload["harnesses"]["custom"] = {"kind": "custom"}

        with pytest.raises(FabricValidationError, match="Fabric config translation failed: Unsupported harness kind"):
            await validate_platform_agent_config(payload, base_dir=Path("/tmp/agent"), fabric=_FakeFabric())

    async def test_fabric_plan_error_is_reported(self, fake_fabric_stack: None) -> None:
        fabric = _FakeFabric(plan_error=_FakeFabricConfigError("bad config"))

        with pytest.raises(FabricValidationError, match="Fabric plan failed: bad config"):
            await validate_platform_agent_config(_example_platform_config(), base_dir=Path("/tmp/agent"), fabric=fabric)

    async def test_preflight_error_is_reported(self, fake_fabric_stack: None) -> None:
        doctor_report = _FakeDoctorReport(
            {
                "status": "fail",
                "checks": [
                    {"name": "requirement.binary", "status": "fail", "message": "binary `codex` missing"},
                ],
            }
        )
        fabric = _FakeFabric(doctor_report=doctor_report)

        with pytest.raises(FabricPreflightError) as error_info:
            await validate_platform_agent_config(_example_platform_config(), base_dir=Path("/tmp/agent"), fabric=fabric)

        assert error_info.value.failed_checks == ["requirement.binary: fail - binary `codex` missing"]
