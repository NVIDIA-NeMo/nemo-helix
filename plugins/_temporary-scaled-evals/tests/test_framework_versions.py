# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework and runtime runner identity resolution."""

from __future__ import annotations

import pytest

try:
    from scaled_evals.api.framework_versions import resolve_framework_runner
    from scaled_evals.api.settings import settings
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_gym_runtime_replaces_launch_artifact_but_preserves_harbor_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_revision = "a" * 40
    image_digest = "sha256:" + "b" * 64
    sbom_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(settings, "gym_runner_image", "registry.example/gym:0.4.0")
    monkeypatch.setattr(settings, "gym_runner_image_digest", image_digest)
    monkeypatch.setattr(settings, "gym_source_revision", source_revision)
    monkeypatch.setattr(settings, "gym_package_version", "0.4.0")
    monkeypatch.setattr(
        settings,
        "gym_runner_image_sbom_ref",
        "oci://registry.example/gym-sbom@sha256:artifact",
    )
    monkeypatch.setattr(settings, "gym_runner_image_sbom_digest", sbom_digest)

    resolved = resolve_framework_runner("harbor", None, runtime="gym_daytona")

    assert resolved.image_ref == "registry.example/gym:0.4.0"
    assert resolved.image_digest == image_digest
    assert resolved.metadata["framework_artifact"]["image_ref"]
    assert resolved.metadata["artifact"] == {
        "kind": "gym-runtime-runner",
        "image_ref": "registry.example/gym:0.4.0",
        "image_digest": image_digest,
        "source_revision": source_revision,
        "package_version": "0.4.0",
        "external_sbom": {
            "ref": "oci://registry.example/gym-sbom@sha256:artifact",
            "digest": sbom_digest,
            "subject_digest": image_digest,
        },
    }
    assert resolved.metadata["gym"]["runtime"] == "gym_daytona"
    assert resolved.metadata["gym"]["provider"] == "daytona"
    assert resolved.metadata["gym"]["identity_completeness"] == "complete"


def test_gym_runtime_rejects_ambiguous_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gym_source_revision", "main")

    with pytest.raises(ValueError, match="full 40- or 64-character git SHA"):
        resolve_framework_runner("nemo_gym", None, runtime="gym_sandbox_daytona")


def test_unknown_gym_identity_is_recorded_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gym_runner_image", "scaled-evals-gym-runner:dev")
    monkeypatch.setattr(settings, "gym_runner_image_digest", None)
    monkeypatch.setattr(settings, "gym_source_revision", "unknown")
    monkeypatch.setattr(settings, "gym_package_version", "unknown")

    resolved = resolve_framework_runner(
        "nemo_gym",
        None,
        runtime="gym_sandbox_opensandbox",
    )

    assert resolved.metadata["artifact"]["source_revision"] is None
    assert resolved.metadata["artifact"]["package_version"] is None
    assert resolved.metadata["gym"]["identity_completeness"] == "incomplete"


def test_digest_pinned_gym_image_supplies_expected_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "d" * 64
    monkeypatch.setattr(settings, "gym_runner_image", f"registry.example/gym@{digest}")
    monkeypatch.setattr(settings, "gym_runner_image_digest", None)
    monkeypatch.setattr(settings, "gym_source_revision", "e" * 40)

    resolved = resolve_framework_runner("harbor", None, runtime="gym_daytona")

    assert resolved.image_digest == digest


def test_hosted_gym_rejects_mutable_runner_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gym_runner_mode", "process")
    monkeypatch.setattr(settings, "gym_runner_image", "registry.example/gym:0.4.0")
    monkeypatch.setattr(settings, "gym_runner_image_digest", None)
    monkeypatch.setattr(settings, "gym_source_revision", "f" * 40)

    with pytest.raises(ValueError, match="immutable GYM_RUNNER_IMAGE_DIGEST"):
        resolve_framework_runner("nemo_gym", None, runtime="gym_sandbox_opensandbox")
