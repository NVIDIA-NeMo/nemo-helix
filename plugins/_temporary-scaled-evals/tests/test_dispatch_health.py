# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tier-3 dispatch readiness probes."""

from __future__ import annotations

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api import dispatch_health
from scaled_evals.api.settings import settings


def test_gym_dispatch_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gym_sandbox_daytona_enabled", False)
    monkeypatch.setattr(settings, "gym_sandbox_opensandbox_enabled", False)
    monkeypatch.setattr(settings, "gym_daytona_enabled", False)
    assert dispatch_health.check_gym_dispatch() == "skipped: disabled"


def test_sandbox_k8s_dispatch_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", False)
    assert dispatch_health.check_sandbox_k8s_dispatch() == "skipped: disabled"


def test_sandbox_k8s_dispatch_configured_mode_reports_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dispatch_health_mode", "configured")
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", True)
    monkeypatch.setattr(settings, "sandbox_k8s_config_path", "/cfg/oracle.yaml")
    monkeypatch.setattr(settings, "sandbox_k8s_env_file", "/run/target/astra.env")
    monkeypatch.setattr(settings, "sandbox_k8s_jobs_dir", "jobs/astra")

    assert dispatch_health.check_sandbox_k8s_dispatch() == "ok"


def test_sandbox_k8s_dispatch_configured_mode_reports_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dispatch_health_mode", "configured")
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", True)
    monkeypatch.setattr(settings, "sandbox_k8s_config_path", "/cfg/oracle.yaml")
    monkeypatch.setattr(settings, "sandbox_k8s_env_file", None)
    monkeypatch.setattr(settings, "sandbox_k8s_jobs_dir", "jobs/astra")

    assert dispatch_health.check_sandbox_k8s_dispatch() == "fail: missing SANDBOX_K8S_ENV_FILE"


def test_sandbox_k8s_process_mode_reports_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    harbor_dir = tmp_path / "harbor"
    kube_dir = tmp_path / ".kube"
    harbor_dir.mkdir()
    kube_dir.mkdir()
    config = tmp_path / "harbor.yaml"
    env_file = tmp_path / "astra.env"
    (kube_dir / "config").write_text("apiVersion: v1\n", encoding="utf-8")
    config.write_text("agents: {}\n", encoding="utf-8")
    env_file.write_text("SANDBOX_NAMESPACE=ns\n", encoding="utf-8")

    monkeypatch.setattr(settings, "dispatch_health_mode", "compose")
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", True)
    monkeypatch.setattr(settings, "sandbox_k8s_config_path", str(config))
    monkeypatch.setattr(settings, "sandbox_k8s_env_file", str(env_file))
    monkeypatch.setattr(settings, "sandbox_k8s_jobs_dir", "jobs/astra")
    monkeypatch.setattr(settings, "harbor_runner_image", "")
    monkeypatch.setattr(settings, "harbor_dir", str(harbor_dir))
    monkeypatch.setattr(settings, "kube_config_dir_host", str(kube_dir))

    assert dispatch_health.check_sandbox_k8s_dispatch() == "ok"


def test_sandbox_k8s_process_mode_reports_missing_harbor_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    kube_dir = tmp_path / ".kube"
    kube_dir.mkdir()
    config = tmp_path / "harbor.yaml"
    env_file = tmp_path / "astra.env"
    (kube_dir / "config").write_text("apiVersion: v1\n", encoding="utf-8")
    config.write_text("agents: {}\n", encoding="utf-8")
    env_file.write_text("SANDBOX_NAMESPACE=ns\n", encoding="utf-8")

    monkeypatch.setattr(settings, "dispatch_health_mode", "compose")
    monkeypatch.setattr(settings, "sandbox_k8s_enabled", True)
    monkeypatch.setattr(settings, "sandbox_k8s_config_path", str(config))
    monkeypatch.setattr(settings, "sandbox_k8s_env_file", str(env_file))
    monkeypatch.setattr(settings, "sandbox_k8s_jobs_dir", "jobs/astra")
    monkeypatch.setattr(settings, "harbor_runner_image", "")
    monkeypatch.setattr(settings, "harbor_dir", str(tmp_path / "missing-harbor"))
    monkeypatch.setattr(settings, "kube_config_dir_host", str(kube_dir))

    assert dispatch_health.check_sandbox_k8s_dispatch().startswith("fail: HARBOR_DIR directory missing:")


def test_hosted_gym_dispatch_reports_incomplete_runner_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dispatch_health_mode", "configured")
    monkeypatch.setattr(settings, "gym_sandbox_opensandbox_enabled", True)
    monkeypatch.setattr(settings, "gym_sandbox_daytona_enabled", False)
    monkeypatch.setattr(settings, "gym_daytona_enabled", False)
    monkeypatch.setattr(settings, "gym_runner_mode", "process")
    monkeypatch.setattr(settings, "gym_sandbox_opensandbox_env_file", "/run/gym/opensandbox.env")
    monkeypatch.setattr(settings, "gym_runner_image", "registry.example/gym:0.4.0")
    monkeypatch.setattr(settings, "gym_runner_image_digest", None)
    monkeypatch.setattr(settings, "gym_source_revision", None)

    assert dispatch_health.check_gym_dispatch() == ("fail: missing GYM_RUNNER_IMAGE_DIGEST, GYM_SOURCE_REVISION")
