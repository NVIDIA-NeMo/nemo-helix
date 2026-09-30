# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run provenance manifest schema and redaction tests."""

from __future__ import annotations

import hashlib
import json

import pytest

try:
    from scaled_evals.models.execution_snapshot import EXECUTION_SNAPSHOT_SCHEMA_VERSION
    from scaled_evals.models.provenance import (
        MANIFEST_SCHEMA_VERSION,
        RunProvenanceManifest,
        build_run_provenance_manifest,
        write_run_provenance_manifest,
    )
    from scaled_evals.models.sbom import SBOM_FILE_NAME
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def _row(**overrides):  # noqa: ANN001, ANN202
    row = {
        "id": "ev_test123",
        "framework": "harbor",
        "requested_framework_version": "stable",
        "framework_version": "0.6.4",
        "framework_adapter_version": "scaled-evals-overlay-v1",
        "sandbox_k8s_version": "0.1.13",
        "runner_metadata": {
            "qualification": {"release": {"version": "0.6.4", "wheel_sha256": "abc"}},
            "artifact": {
                "image_ref": "registry.example/runner:0.6.4",
                "image_digest": "sha256:resolved",
                "source_revision": "runner-sha",
                "signature_ref": "oci://registry.example/runner:signature",
                "signature_digest": "sha256:signed",
                "signature_audit_id": "audit-123",
            },
            "agent_bundle": {
                "bundle_id": "ab_claude",
                "bundle_name": "claude-code-stable",
                "agent_name": "claude-code",
                "agent_version": "2.1.133",
                "image_ref": "registry.example/claude:2.1.133",
                "image_digest": "registry.example/claude@sha256:" + "d" * 64,
                "entrypoint": "bin/claude",
                "source_lock_digest": "sha256:" + "e" * 64,
                "fingerprint": "sha256:" + "f" * 64,
            },
        },
        "runner_image_ref": "registry.example/runner@sha256:resolved",
        "runner_image_digest": "sha256:resolved",
        "task_id": "task_abc",
        "task_revision": 2,
        "image_ref": "registry.example/task_abc:rev2",
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "framework_profile_id": "cfg_framework",
        "harbor_profile_id": "cfg_harbor",
        "intake_profile_id": "cfg_intake",
        "credentials": {"openai": "cred_openai"},
        "harbor_config": {
            "env": {
                "SAFE_PROFILE": "dev",
                "POLICY_API_KEY": "sk-do-not-serialize",
            }
        },
        "intake_config": {
            "endpoint": "https://intake.example.test/apis/intake/v2",
            "workspace": "team-ws",
            "source": "scaled-evals",
            "app": "bench-app",
            "task": "trial-task",
            "experiment_id": "exp-1",
        },
    }
    row.update(overrides)
    return row


def test_run_provenance_manifest_schema_round_trips(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("CI_COMMIT_SHA", "abc123")
    monkeypatch.setenv("CI_PIPELINE_ID", "456")
    monkeypatch.setenv("HARBOR_RUNNER_IMAGE", "registry.example/runner:latest")
    monkeypatch.setenv("HARBOR_RUNNER_IMAGE_DIGEST", "sha256:runner")
    monkeypatch.setenv("API_IMAGE_SBOM_REF", "oci://registry.example/api@sha256:sbom")
    monkeypatch.setenv("API_IMAGE_SBOM_DIGEST", "sha256:sbom")

    manifest = build_run_provenance_manifest(
        _row(),
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
        artifact_root=tmp_path,
        backend="fake",
        handle="handle-123",
    )

    data = json.loads(manifest.model_dump_json(exclude_none=True))
    parsed = RunProvenanceManifest.model_validate(data)
    assert parsed.schema_version == MANIFEST_SCHEMA_VERSION
    assert parsed.control_plane.git_sha == "abc123"
    assert parsed.runtime.network_policy == "unrestricted"
    assert parsed.runtime.effective_isolation.direct_egress == "unrestricted"
    assert parsed.runtime.effective_isolation.model_gateway == "none"
    assert parsed.runtime.effective_isolation.platform_verified is False
    assert parsed.runtime.runner_image_digest == "sha256:resolved"
    assert parsed.runtime.runner_image_ref == "registry.example/runner@sha256:resolved"
    assert parsed.config.requested_framework_version == "stable"
    assert parsed.config.framework_version == "0.6.4"
    assert parsed.config.framework_adapter_version == "scaled-evals-overlay-v1"
    assert parsed.config.sandbox_k8s_version == "0.1.13"
    assert parsed.config.runner_metadata["artifact"]["signature_digest"] == "sha256:signed"
    assert parsed.harbor is not None
    assert parsed.harbor.profile_id == "cfg_harbor"
    assert parsed.harbor.requested_version == "stable"
    assert parsed.harbor.version == "0.6.4"
    assert parsed.harbor.runner_source_revision == "runner-sha"
    assert parsed.harbor.signature_ref == "oci://registry.example/runner:signature"
    assert parsed.harbor.signature_digest == "sha256:signed"
    assert parsed.harbor.signature_audit_id == "audit-123"
    assert parsed.harbor.profile_config_hash is not None
    assert parsed.intake is not None
    assert parsed.intake.profile_id == "cfg_intake"
    assert parsed.intake.endpoint == "https://intake.example.test/apis/intake/v2"
    assert parsed.intake.workspace == "team-ws"
    assert parsed.intake.experiment_id == "exp-1"
    assert parsed.intake.profile_config_hash is not None
    agent_bundle = parsed.config.runner_metadata["agent_bundle"]
    assert agent_bundle["bundle_id"] == "ab_claude"
    assert agent_bundle["bundle_name"] == "claude-code-stable"
    assert agent_bundle["agent_name"] == "claude-code"
    assert agent_bundle["agent_version"] == "2.1.133"
    assert agent_bundle["image_ref"] == "registry.example/claude:2.1.133"
    assert agent_bundle["image_digest"].endswith("d" * 64)
    assert parsed.artifacts.provenance_manifest_path == "scaled-evals-provenance.json"
    assert parsed.sbom[0].name == "API_IMAGE_SBOM_DIGEST"
    assert parsed.sbom[0].digest == "sha256:sbom"
    assert parsed.sbom[1].name == "API_IMAGE_SBOM_REF"
    assert parsed.sbom[1].value == "oci://registry.example/api@sha256:sbom"


def test_run_provenance_records_managed_harbor_dataset_images(tmp_path) -> None:  # noqa: ANN001
    source_digest = "sha256:" + "a" * 64
    destination_digest = "sha256:" + "b" * 64
    imported = {
        "source_image": "alexgshaw/task:20251031",
        "source_immutable_image": f"docker.io/alexgshaw/task@{source_digest}",
        "task_id": "task_import",
        "task_revision": 1,
        "image_ref": "registry.example/task_import:rev1",
        "image_digest": destination_digest,
        "runtime_image": f"registry.example/task_import@{destination_digest}",
    }
    provenance_path = write_run_provenance_manifest(
        tmp_path,
        _row(
            backend_handle={
                "backend": "sandbox_k8s",
                "external_id": "ev_test123",
                "raw": {"harbor_dataset_image_imports": [imported]},
            }
        ),
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
    )

    provenance = json.loads(provenance_path.read_text())
    assert provenance["harbor"]["dataset_images"] == [imported]
    bom = json.loads((tmp_path / SBOM_FILE_NAME).read_text())
    component = next(
        item for item in bom["components"] if item["bom-ref"] == "urn:scaled-evals:harbor-dataset-image:ev_test123:0"
    )
    assert component["hashes"] == [{"alg": "SHA-256", "content": "b" * 64}]
    assert {
        "name": "scaled-evals:image:upstream-immutable-reference",
        "value": imported["source_immutable_image"],
    } in component["properties"]


def test_run_provenance_manifest_redacts_secret_values(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DAYTONA_API_KEY", "sk-env-secret")
    monkeypatch.setenv("POLICY_API_KEY", "sk-policy-secret")

    manifest = build_run_provenance_manifest(
        _row(credentials={"openai": "cred_openai"}),
        status="failed",
        artifact_prefix="evaluations/ev_test123/artifacts/",
        artifact_root=tmp_path,
    )

    payload = manifest.model_dump_json(exclude_none=True)
    assert "sk-do-not-serialize" not in payload
    assert "sk-env-secret" not in payload
    assert "sk-policy-secret" not in payload
    assert "cred_openai" in payload
    assert manifest.credentials[0].fingerprint
    assert "harbor_config" in manifest.config.config_hashes
    assert "DAYTONA_API_KEY" not in manifest.environment.env_hashes


def test_write_run_provenance_manifest_propagates_write_errors(tmp_path) -> None:
    blocked_root = tmp_path / "not-a-directory"
    blocked_root.write_text("occupied")

    with pytest.raises(FileExistsError):
        write_run_provenance_manifest(
            blocked_root,
            _row(),
            status="failed",
            artifact_prefix="evaluations/ev_test123/artifacts/",
        )


def test_write_run_provenance_binds_incomplete_cyclonedx_bom(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("TASK_IMAGE_SBOM_REF", "oci://registry.example/task-sbom")
    monkeypatch.setenv("TASK_IMAGE_SBOM_DIGEST", "sha256:sbom-artifact")
    monkeypatch.setenv("TASK_IMAGE_DIGEST", "sha256:" + "a" * 64)
    provenance_path = write_run_provenance_manifest(
        tmp_path,
        _row(
            image_digest="sha256:" + "a" * 64,
            finished_at="2026-07-10T12:00:00+00:00",
        ),
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
    )

    bom_path = tmp_path / SBOM_FILE_NAME
    provenance = json.loads(provenance_path.read_text())
    bom = json.loads(bom_path.read_text())
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.6"
    assert bom["compositions"] == [
        {
            "aggregate": "incomplete",
            "assemblies": ["urn:scaled-evals:evaluation:ev_test123"],
        }
    ]
    task = next(component for component in bom["components"] if component["name"] == "task:task_abc")
    assert {
        "name": "scaled-evals:image:identity-status",
        "value": "declared-unverified",
    } in task["properties"]
    assert task["externalReferences"] == [
        {
            "type": "bom",
            "url": "oci://registry.example/task-sbom",
            "comment": (
                f"artifact_digest=sha256:sbom-artifact; subject_digest=sha256:{'a' * 64}; binding=subject-bound"
            ),
        }
    ]
    expected_digest = "sha256:" + hashlib.sha256(bom_path.read_bytes()).hexdigest()
    assert provenance["run_sbom"] == {
        "name": "cyclonedx-run-composition",
        "value": SBOM_FILE_NAME,
        "digest": expected_digest,
    }


def test_gym_identity_and_external_sbom_come_from_execution_snapshot(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    source_revision = "a" * 40
    runner_digest = "sha256:" + "b" * 64
    sbom_digest = "sha256:" + "c" * 64
    runner_metadata = {
        "framework_artifact": {
            "image_ref": "registry.example/harbor:0.13.2",
            "image_digest": "sha256:" + "d" * 64,
            "source_revision": "e" * 40,
        },
        "artifact": {
            "kind": "gym-runtime-runner",
            "image_ref": "registry.example/gym:0.4.0",
            "image_digest": runner_digest,
            "source_revision": source_revision,
            "package_version": "0.4.0",
            "external_sbom": {
                "ref": "oci://registry.example/gym-sbom@sha256:artifact",
                "digest": sbom_digest,
                "subject_digest": runner_digest,
            },
        },
        "gym": {
            "runtime": "gym_daytona",
            "provider": "daytona",
            "agent_path": "harbor_agent",
            "package_version": "0.4.0",
            "source_revision": source_revision,
            "identity_completeness": "complete",
            "identity_verification": "declared-unverified",
            "external_sbom": {
                "ref": "oci://registry.example/gym-sbom@sha256:artifact",
                "digest": sbom_digest,
                "subject_digest": runner_digest,
            },
        },
    }
    snapshot = {
        "schema_version": "scaled-evals-execution-inputs-v1",
        "captured_at": "2026-07-10T12:00:00+00:00",
        "evaluation": {
            "framework": "harbor",
            "framework_version": "0.13.2",
            "framework_profile_id": "cfg_gym",
            "runner_image_ref": "registry.example/gym:0.4.0",
            "runner_image_digest": runner_digest,
            "runner_metadata": runner_metadata,
            "runtime": "gym_daytona",
        },
        "task": {},
        "profiles": {},
        "credentials": {},
        "submission_identity": {},
    }
    monkeypatch.setenv("GYM_RUNNER_IMAGE", "mutable-live:latest")
    monkeypatch.setenv("GYM_RUNNER_IMAGE_DIGEST", "sha256:" + "f" * 64)
    monkeypatch.setenv("GYM_RUNNER_IMAGE_SBOM_REF", "oci://mutable/live-sbom")

    provenance_path = write_run_provenance_manifest(
        tmp_path,
        _row(
            runtime="gym_daytona",
            framework_profile_id="cfg_gym",
            runner_image_ref="mutable-row:latest",
            runner_image_digest=None,
            runner_metadata={},
            execution_snapshot=snapshot,
            backend_handle={
                "backend": "gym_daytona",
                "external_id": "ev_test123",
                "raw": {
                    "observed_runner_image_digest": runner_digest,
                    "observed_runner_image_id": "sha256:" + "1" * 64,
                    "observed_gym_source_revision": source_revision,
                    "observed_gym_package_version": "0.4.0",
                },
            },
            finished_at="2026-07-10T12:00:00+00:00",
        ),
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
    )

    provenance = json.loads(provenance_path.read_text())
    gym = provenance["gym"]
    assert gym["runtime"] == "gym_daytona"
    assert gym["provider"] == "daytona"
    assert gym["agent_path"] == "harbor_agent"
    assert gym["profile_id"] == "cfg_gym"
    assert gym["package_version"] == "0.4.0"
    assert gym["source_revision"] == source_revision
    assert gym["runner_image_ref"] == "registry.example/gym:0.4.0"
    assert gym["runner_image_digest"] == runner_digest
    assert gym["observed_runner_image_digest"] == runner_digest
    assert gym["identity_completeness"] == "complete"
    assert gym["identity_verification"] == "runtime-observed"
    assert provenance["runtime"]["runner_image_ref"] == "registry.example/gym:0.4.0"
    assert provenance["runtime"]["sandbox"] == {}
    assert provenance["harbor"]["runner_image_ref"] == "registry.example/harbor:0.13.2"
    assert "mutable-live:latest" not in provenance_path.read_text()
    assert "oci://mutable/live-sbom" not in provenance_path.read_text()

    bom = json.loads((tmp_path / SBOM_FILE_NAME).read_text())
    runner = next(component for component in bom["components"] if component["name"] == "nemo-gym-runtime-runner")
    assert runner["version"] == "0.4.0"
    assert {
        "name": "scaled-evals:image:identity-status",
        "value": "runtime-observed",
    } in runner["properties"]
    assert runner["externalReferences"] == [
        {
            "type": "bom",
            "url": "oci://registry.example/gym-sbom@sha256:artifact",
            "comment": (f"artifact_digest={sbom_digest}; subject_digest={runner_digest}; binding=subject-bound"),
        }
    ]
    assert {
        "name": "scaled-evals:external-sbom:GYM_RUNNER_IMAGE:binding",
        "value": "subject-bound",
    } in bom["metadata"]["properties"]


def test_gym_provenance_projects_validated_profile_and_allowlisted_executor(tmp_path) -> None:
    profile_config = {
        "schema_version": "1",
        "command": "run_and_collect",
        "config_paths": [
            "/harness/gym-sandbox-opensandbox/configs/mini_swe_agent_opensandbox_smoke.yaml",
            "responses_api_models/openai_model/configs/openai_model.yaml",
        ],
        "agent_name": "mini_swe_agent_2",
        "split": "validation",
        "limit": 3,
        "num_repeats": 2,
        "num_samples_in_parallel": 4,
        "overrides": {"skip_venv_if_present": True},
    }
    snapshot = {
        "schema_version": "scaled-evals-execution-inputs-v1",
        "captured_at": "2026-07-13T12:00:00+00:00",
        "evaluation": {
            "framework": "nemo_gym",
            "framework_profile_id": "cfg_gym",
            "runtime": "gym_sandbox_opensandbox",
            "runner_metadata": {},
        },
        "task": {},
        "profiles": {"framework": {"id": "cfg_gym", "type": "gym", "config": profile_config}},
        "credentials": {},
        "submission_identity": {},
    }
    path = write_run_provenance_manifest(
        tmp_path,
        _row(
            framework="nemo_gym",
            runtime="gym_sandbox_opensandbox",
            framework_profile_id="cfg_gym",
            execution_snapshot=snapshot,
            framework_config={"command": "live-config-must-not-win"},
            n_attempts=2,
            parallelism=6,
            dispatch_job_name="scaled-evals-eval-test123",
            dispatch_job_uid="job-uid-123",
            backend_handle={
                "backend": "gym_sandbox_opensandbox",
                "external_id": "ev_test123",
                "raw": {
                    "command": "run_and_collect",
                    "process": True,
                    "process_owner_pod": "eval-runner-pod-abc",
                    "process_pid": 42,
                    "process_start_identity": "unsafe-start-id",
                    "exit_code_path": "/tmp/private/exit-code",
                    "argv": ["--policy-api-key=sk-do-not-serialize"],
                },
            },
        ),
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    profile = manifest["gym"]["profile"]
    executor = manifest["gym"]["executor"]
    assert profile == {
        "source": "snapshot",
        "projection": "validated-v1",
        "schema_version": "1",
        "config_sha256": manifest["execution_inputs"]["profiles"]["framework"]["config_sha256"],
        "requested_command": "run_and_collect",
        "observed_command": "run_and_collect",
        "command_verification": "matched",
        "config_paths": profile_config["config_paths"],
        "requested_split": "validation",
        "requested_limit": 3,
        "effective_limit": 3,
        "requested_num_repeats": 2,
        "requested_num_samples_in_parallel": 4,
        "effective_num_samples_in_parallel": 4,
        "control_plane_parallelism": 6,
        "control_plane_attempts": 2,
    }
    assert executor == {
        "mode": "process",
        "dispatch_job_name": "scaled-evals-eval-test123",
        "dispatch_job_uid": "job-uid-123",
        "runner_pod_name": "eval-runner-pod-abc",
        "runner_pod_name_source": "backend-handle",
    }
    serialized = path.read_text(encoding="utf-8")
    for forbidden in (
        "sk-do-not-serialize",
        "process_pid",
        "unsafe-start-id",
        "exit_code_path",
        "/tmp/private",
        "argv",
        "live-config-must-not-win",
    ):
        assert forbidden not in serialized


def test_gym_provenance_keeps_invalid_legacy_profile_hash_only(tmp_path) -> None:
    snapshot = {
        "schema_version": "scaled-evals-execution-inputs-v1",
        "captured_at": "2026-07-13T12:00:00+00:00",
        "evaluation": {
            "framework": "nemo_gym",
            "framework_profile_id": "cfg_legacy",
            "runtime": "gym_sandbox_opensandbox",
            "runner_metadata": {},
        },
        "task": {},
        "profiles": {
            "framework": {
                "id": "cfg_legacy",
                "type": "gym",
                "config": {
                    "command": "legacy",
                    "api_key": "sk-legacy-must-not-serialize",
                    "agent_name": "sk-agent-secret-must-not-serialize",
                    "config_paths": ["/private/legacy/path"],
                },
            }
        },
        "credentials": {},
        "submission_identity": {},
    }
    path = write_run_provenance_manifest(
        tmp_path,
        _row(
            framework="nemo_gym",
            runtime="gym_sandbox_opensandbox",
            framework_profile_id="cfg_legacy",
            execution_snapshot=snapshot,
            backend_handle={
                "backend": "gym_sandbox_opensandbox",
                "external_id": "ev_test123",
                "raw": {"command": "sk-observed-secret-must-not-serialize", "process": True},
            },
        ),
        status="failed",
        artifact_prefix="evaluations/ev_test123/artifacts/",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["gym"]["profile"] == {
        "source": "snapshot",
        "projection": "legacy-unparsed",
        "config_sha256": manifest["execution_inputs"]["profiles"]["framework"]["config_sha256"],
    }
    assert manifest["gym"]["agent_path"] == "mini_swe_agent_2"
    serialized = path.read_text(encoding="utf-8")
    assert "sk-legacy-must-not-serialize" not in serialized
    assert "sk-observed-secret-must-not-serialize" not in serialized
    assert "sk-agent-secret-must-not-serialize" not in serialized
    assert "/private/legacy/path" not in serialized


def test_run_provenance_manifest_reports_variant_operational_overrides(tmp_path) -> None:  # noqa: ANN001
    """A variant run shows its frozen policy and the budget the agent really had."""
    row = _row(
        execution_snapshot={
            "schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
            "evaluation": {
                "id": "ev_test123",
                "benchmark_variant": {
                    "derived_from": {"benchmark_id": "bm_base", "revision": 2},
                    "operational_policy": {"agent_timeout_floor_sec": 7200},
                },
            },
            "task": {},
            "profiles": {},
            "credentials": {},
            "submission_identity": {},
        },
        backend_handle={
            "backend": "sandbox_k8s",
            "raw": {"agent_timeout_apply": {"original": 900.0, "effective": 7200.0}},
        },
    )
    manifest = build_run_provenance_manifest(
        row,
        status="succeeded",
        artifact_prefix="evaluations/ev_test123/artifacts/",
        artifact_root=tmp_path,
    )
    assert manifest.operational_overrides == {
        "derived_from": {"benchmark_id": "bm_base", "revision": 2},
        "policy": {"agent_timeout_floor_sec": 7200},
        "agent_timeout_sec": {"original": 900.0, "effective": 7200.0},
    }
    # A plain (non-variant) run carries no override block at all.
    assert (
        build_run_provenance_manifest(
            _row(),
            status="succeeded",
            artifact_prefix="evaluations/ev_test123/artifacts/",
            artifact_root=tmp_path,
        ).operational_overrides
        == {}
    )
