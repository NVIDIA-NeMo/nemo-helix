# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.models.execution_snapshot import (
    canonical_sha256,
    current_process_identity,
    public_execution_projection,
    validate_execution_snapshot,
)
from scaled_evals.models.gym_identity import (
    backend_handle_raw,
    gym_run_identity,
    snapshot_evaluation,
)
from scaled_evals.models.sbom import SBOM_FILE_NAME, file_sha256, write_run_sbom

MANIFEST_SCHEMA_VERSION = "scaled-evals-run-provenance-v2"
MANIFEST_FILE_NAME = "scaled-evals-provenance.json"
_SECRET_KEYWORDS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
    "encrypted_payload",
    "byok",
)
_CONTROL_PLANE_ENV = (
    "CI_COMMIT_SHA",
    "CI_COMMIT_BRANCH",
    "CI_COMMIT_REF_NAME",
    "CI_MERGE_REQUEST_IID",
    "CI_PIPELINE_ID",
    "CI_JOB_ID",
    "CI_REGISTRY_IMAGE",
    "CI_APPLICATION_REPOSITORY",
    "CI_APPLICATION_TAG",
    "CONTROL_PLANE_IMAGE",
    "CONTROL_PLANE_IMAGE_DIGEST",
    "API_IMAGE",
    "API_IMAGE_DIGEST",
)
_RUNTIME_ENV = (
    "GYM_RUNNER_IMAGE",
    "GYM_RUNNER_IMAGE_DIGEST",
    "HARBOR_RUNNER_IMAGE",
    "HARBOR_RUNNER_IMAGE_DIGEST",
    "SANDBOX_K8S_TARGET",
    "SANDBOX_K8S_CLUSTER",
    "SANDBOX_K8S_PROFILE",
    "DAYTONA_ORGANIZATION_ID",
    "DAYTONA_WORKSPACE_ID",
    "OPEN_SANDBOX_CELL_ID",
    "OPEN_SANDBOX_CLUSTER",
)


class ProvenanceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | None = None
    digest: str | None = None


class TaskIdentifierProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_revision: int
    task_slug: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    image_tag: str | None = None
    tarball_sha256: str | None = None
    tarball_object_key: str | None = None


class AgentIdentifierProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str | None = None
    bundle_name: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    image_tag: str | None = None
    source_lock_digest: str | None = None
    fingerprint: str | None = None
    entrypoint: str | None = None
    platform: str | None = None
    runtime_abi: str | None = None
    bundle_layout_version: int | None = None
    builder_profile: str | None = None


class ControlPlaneProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    git_sha: str | None = None
    git_branch: str | None = None
    merge_request: str | None = None
    ci_pipeline_id: str | None = None
    ci_job_id: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    package_version: str | None = None
    release_version: str | None = None
    signature_ref: str | None = None
    signature_digest: str | None = None
    signature_audit_id: str | None = None
    identity_observation: str = "declared"


class TaskProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    revision: int
    slug: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    image_tag: str | None = None
    tarball_sha256: str | None = None
    tarball_object_key: str | None = None
    identifiers: TaskIdentifierProvenance


class EffectiveIsolationProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direct_egress: str
    model_gateway: str
    bypass_resistant: bool | None = None
    platform_verified: bool = False
    warnings: list[str] = Field(default_factory=list)


class RuntimeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    network_policy: str = "unrestricted"
    effective_isolation: EffectiveIsolationProvenance
    backend: str | None = None
    handle: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    sandbox: dict[str, str] = Field(default_factory=dict)


class ConfigProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    requested_framework_version: str | None = None
    framework_version: str | None = None
    framework_adapter_version: str | None = None
    sandbox_k8s_version: str | None = None
    runner_metadata: dict[str, Any] = Field(default_factory=dict)
    framework_profile_id: str | None = None
    harbor_profile_id: str | None = None
    switchyard_profile_id: str | None = None
    intake_profile_id: str | None = None
    agent_bundle: AgentIdentifierProvenance | None = None
    config_hashes: dict[str, str] = Field(default_factory=dict)


class HarborDatasetImageProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_image: str
    source_immutable_image: str
    task_id: str
    task_revision: int
    image_ref: str
    image_digest: str
    runtime_image: str


class HarborProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str | None = None
    requested_version: str | None = None
    version: str | None = None
    git_commit_sha: str | None = None
    wheel_sha256: str | None = None
    adapter_version: str | None = None
    adapter_bundle_sha256: str | None = None
    sandbox_k8s_version: str | None = None
    sandbox_k8s_git_commit_sha: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    runner_image_tag: str | None = None
    runner_source_revision: str | None = None
    runner_ci_pipeline_id: str | None = None
    runner_ci_job_id: str | None = None
    signature_ref: str | None = None
    signature_digest: str | None = None
    signature_audit_id: str | None = None
    profile_config_hash: str | None = None
    dataset_images: list[HarborDatasetImageProvenance] = Field(default_factory=list)


class GymProfileProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["snapshot", "live-legacy", "absent"]
    projection: Literal["validated-v1", "legacy-unparsed", "absent"]
    schema_version: str | None = None
    config_sha256: str | None = None
    requested_command: str | None = None
    observed_command: str | None = None
    command_verification: Literal["matched", "mismatch", "unobserved"] | None = None
    config_paths: list[str] | None = None
    requested_input_jsonl_fpath: str | None = None
    requested_split: str | None = None
    requested_limit: int | None = None
    effective_limit: int | None = None
    requested_num_repeats: int | None = None
    requested_num_samples_in_parallel: int | None = None
    effective_num_samples_in_parallel: int | None = None
    control_plane_parallelism: int | None = None
    control_plane_attempts: int | None = None


class GymExecutorProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["process", "docker", "unknown"]
    dispatch_job_name: str | None = None
    dispatch_job_uid: str | None = None
    runner_pod_name: str | None = None
    runner_pod_name_source: Literal["backend-handle"] | None = None


class GymProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: str
    provider: str
    agent_path: str
    framework: str | None = None
    profile_id: str | None = None
    package_version: str | None = None
    source_revision: str | None = None
    runner_image_ref: str | None = None
    runner_image_digest: str | None = None
    observed_runner_image_digest: str | None = None
    observed_runner_image_id: str | None = None
    observed_source_revision: str | None = None
    observed_package_version: str | None = None
    identity_completeness: Literal["complete", "incomplete"]
    identity_verification: Literal["declared-unverified", "runtime-observed", "mismatch"]
    external_sbom: dict[str, str] = Field(default_factory=dict)
    profile: GymProfileProvenance
    executor: GymExecutorProvenance
    runtime_settings: dict[str, Any] = Field(default_factory=dict)


class IntakeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str | None = None
    endpoint: str | None = None
    workspace: str | None = None
    source: str | None = None
    app: str | None = None
    task: str | None = None
    experiment_id: str | None = None
    profile_config_hash: str | None = None


class CredentialProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    credential_id: str
    fingerprint: str
    provider: str | None = None
    payload_kind: str | None = None
    fingerprint_source: str = "stored-credential"


class EnvironmentProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_hashes: dict[str, str] = Field(default_factory=dict)


class ToolProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str
    platform: str
    package_lock_hash: str | None = None
    tools: dict[str, str] = Field(default_factory=dict)


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_store_prefix: str
    artifact_manifest_path: str
    provenance_manifest_path: str
    sbom_path: str


class SwitchyardProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str | None = None
    topology: Literal["shared_campaign", "dedicated", "dedicated_retry"] | None = None
    mode: str = "managed"
    namespace: str | None = None
    deployment: str | None = None
    service: str | None = None
    config_map: str | None = None
    secret: str | None = None
    network_policy: str | None = None
    endpoint: str
    openai_base_url: str | None = None
    anthropic_base_url: str | None = None
    inbound: str | None = None
    book_mode: str | None = None
    port: int | None = None
    manifest_hash: str | None = None
    config_hash: str | None = None
    drain_until: str | None = None
    artifact_path: str | None = None
    endpoint_identity: str | None = None
    trust_warning: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    source_project: str | None = None
    source_ref: str | None = None
    source_commit: str | None = None
    context_path: str | None = None
    dockerfile_path: str | None = None
    dockerfile_sha256: str | None = None
    context_hash: str | None = None


class RunProvenanceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scaled-evals-run-provenance-v2"] = MANIFEST_SCHEMA_VERSION
    generated_at: datetime
    evaluation_id: str
    status: str
    control_plane: ControlPlaneProvenance = Field(default_factory=ControlPlaneProvenance)
    task: TaskProvenance
    runtime: RuntimeProvenance
    config: ConfigProvenance
    credentials: list[CredentialProvenance] = Field(default_factory=list)
    environment: EnvironmentProvenance = Field(default_factory=EnvironmentProvenance)
    tools: ToolProvenance
    artifacts: ArtifactProvenance
    harbor: HarborProvenance | None = None
    gym: GymProvenance | None = None
    switchyard: SwitchyardProvenance | None = None
    intake: IntakeProvenance | None = None
    execution_inputs: dict[str, Any] = Field(default_factory=dict)
    execution_identity: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)
    operational_overrides: dict[str, Any] = Field(default_factory=dict)
    run_sbom: ProvenanceReference
    sbom: list[ProvenanceReference] = Field(default_factory=list)
    identifiers: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def build_run_provenance_manifest(
    row: Mapping[str, Any],
    *,
    status: str,
    artifact_prefix: str,
    artifact_root: Path,
    backend: str | None = None,
    handle: str | None = None,
    run_sbom_digest: str | None = None,
) -> RunProvenanceManifest:
    task_identifiers = _task_identifiers(row)
    agent_bundle = _agent_bundle(row)
    snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
    evaluation = snapshot_evaluation(row)
    runner_metadata = _mapping(evaluation.get("runner_metadata")) or _mapping(row.get("runner_metadata"))
    execution_inputs = {"legacy_live_lookup": True} if snapshot is None else public_execution_projection(snapshot)
    if row.get("extra_skill_materials"):
        execution_inputs["observed_extra_skill_materials"] = _redacted_value(row["extra_skill_materials"])
    return RunProvenanceManifest(
        generated_at=_generated_at(row),
        evaluation_id=str(row["id"]),
        status=status,
        control_plane=_control_plane(),
        task=TaskProvenance(
            id=str(row["task_id"]),
            revision=int(row["task_revision"]),
            slug=task_identifiers.task_slug,
            image_ref=_clean_optional(row.get("image_ref")),
            image_digest=_clean_optional(row.get("image_digest") or row.get("task_image_digest")),
            image_tag=task_identifiers.image_tag,
            tarball_sha256=_clean_optional(row.get("tarball_sha256")),
            tarball_object_key=_clean_optional(row.get("tarball_object_key")),
            identifiers=task_identifiers,
        ),
        runtime=_runtime(row, backend=backend, handle=handle),
        config=ConfigProvenance(
            framework=str(evaluation.get("framework") or row["framework"]),
            requested_framework_version=_clean_optional(
                evaluation.get("requested_framework_version") or row.get("requested_framework_version")
            ),
            framework_version=_clean_optional(evaluation.get("framework_version") or row.get("framework_version")),
            framework_adapter_version=_clean_optional(
                evaluation.get("framework_adapter_version") or row.get("framework_adapter_version")
            ),
            sandbox_k8s_version=_clean_optional(
                evaluation.get("sandbox_k8s_version") or row.get("sandbox_k8s_version")
            ),
            runner_metadata=_redacted_value(dict(runner_metadata)),
            framework_profile_id=_clean_optional(row.get("framework_profile_id")),
            harbor_profile_id=_clean_optional(row.get("harbor_profile_id")),
            switchyard_profile_id=_clean_optional(row.get("switchyard_profile_id")),
            intake_profile_id=_clean_optional(row.get("intake_profile_id")),
            agent_bundle=agent_bundle,
            config_hashes=_config_hashes(row),
        ),
        credentials=_credential_refs(row.get("credentials") or {}, snapshot=snapshot),
        environment=EnvironmentProvenance(env_hashes=_env_hashes()),
        tools=ToolProvenance(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            package_lock_hash=_file_hash(Path("/app/uv.lock")) or _file_hash(Path("uv.lock")),
            tools=_tool_versions(("uv", "ruff", "pytest")),
        ),
        artifacts=ArtifactProvenance(
            object_store_prefix=artifact_prefix,
            artifact_manifest_path="scaled-evals-manifest.json",
            provenance_manifest_path=MANIFEST_FILE_NAME,
            sbom_path=SBOM_FILE_NAME,
        ),
        harbor=_harbor(row),
        gym=_gym(row),
        switchyard=_switchyard(row),
        intake=_intake(row),
        execution_inputs=execution_inputs,
        execution_identity=current_process_identity(),
        outcome=_outcome(row, status=status),
        operational_overrides=_operational_overrides(row),
        run_sbom=ProvenanceReference(
            name="cyclonedx-run-composition",
            value=SBOM_FILE_NAME,
            digest=run_sbom_digest,
        ),
        sbom=_sbom_refs(row),
        identifiers=_identifier_summary(
            row,
            task_identifiers=task_identifiers,
            agent_bundle=agent_bundle,
        ),
        notes=_manifest_notes(artifact_root, row),
    )


def _operational_overrides(row: Mapping[str, Any]) -> dict[str, Any]:
    """Report a benchmark variant's operational overrides and what they changed.

    ``policy`` is the floor frozen at submission; ``agent_timeout_sec`` is the
    original-to-effective diff the dispatcher actually wrote into the staged
    ``task.toml``, so a score of record shows the budget the agent really had.
    """
    variant = _mapping(snapshot_evaluation(row).get("benchmark_variant"))
    applied = _mapping(backend_handle_raw(row.get("backend_handle")).get("agent_timeout_apply"))
    overrides: dict[str, Any] = {}
    if variant.get("derived_from"):
        overrides["derived_from"] = _redacted_value(variant["derived_from"])
    if variant.get("operational_policy"):
        overrides["policy"] = _redacted_value(variant["operational_policy"])
    if applied:
        overrides["agent_timeout_sec"] = _redacted_value(dict(applied))
    return overrides


def write_run_provenance_manifest(
    root: Path | str,
    row: Mapping[str, Any],
    *,
    status: str,
    artifact_prefix: str,
    backend: str | None = None,
    handle: str | None = None,
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    sbom_path = write_run_sbom(root_path, row)
    manifest = build_run_provenance_manifest(
        row,
        status=status,
        artifact_prefix=artifact_prefix,
        artifact_root=root_path,
        backend=backend,
        handle=handle,
        run_sbom_digest=file_sha256(sbom_path),
    )
    path = root_path / MANIFEST_FILE_NAME
    path.write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    return path


def _control_plane() -> ControlPlaneProvenance:
    identity = current_process_identity()
    return ControlPlaneProvenance(
        git_sha=identity.get("source_revision") or _first_env("CI_COMMIT_SHA", "GIT_COMMIT", "GIT_SHA"),
        git_branch=_first_env("CI_COMMIT_BRANCH", "CI_COMMIT_REF_NAME", "GIT_BRANCH"),
        merge_request=_first_env("CI_MERGE_REQUEST_IID", "CI_MERGE_REQUEST_ID"),
        ci_pipeline_id=identity.get("ci_pipeline_id") or _first_env("CI_PIPELINE_ID"),
        ci_job_id=identity.get("ci_job_id") or _first_env("CI_JOB_ID"),
        image_ref=identity.get("image_ref")
        or _first_env("CONTROL_PLANE_IMAGE", "API_IMAGE", "CI_APPLICATION_REPOSITORY"),
        image_digest=identity.get("image_digest") or _first_env("CONTROL_PLANE_IMAGE_DIGEST", "API_IMAGE_DIGEST"),
        package_version=identity.get("package_version"),
        release_version=identity.get("release_version"),
        signature_ref=identity.get("signature_ref"),
        signature_digest=identity.get("signature_digest"),
        signature_audit_id=identity.get("signature_audit_id"),
    )


def _runtime(row: Mapping[str, Any], *, backend: str | None, handle: str | None) -> RuntimeProvenance:
    evaluation = snapshot_evaluation(row)
    runtime = str(evaluation.get("runtime") or row["runtime"])
    runner_image_ref = _clean_optional(evaluation.get("runner_image_ref"))
    runner_image_digest = _clean_optional(evaluation.get("runner_image_digest"))
    if row.get("execution_snapshot") is None:
        runner_image_ref = runner_image_ref or _first_env(f"{runtime.upper()}_RUNNER_IMAGE")
        runner_image_digest = runner_image_digest or _first_env(f"{runtime.upper()}_RUNNER_IMAGE_DIGEST")
        if runtime.startswith("gym"):
            runner_image_ref = runner_image_ref or _first_env("GYM_RUNNER_IMAGE")
            runner_image_digest = runner_image_digest or _first_env("GYM_RUNNER_IMAGE_DIGEST")
    if runtime in ("sandbox_k8s", "harbor_opensandbox"):
        runner_image_ref = runner_image_ref or _first_env("HARBOR_RUNNER_IMAGE")
        runner_image_digest = runner_image_digest or _first_env("HARBOR_RUNNER_IMAGE_DIGEST")
    return RuntimeProvenance(
        name=runtime,
        network_policy=str(row.get("network_policy") or "unrestricted"),
        effective_isolation=_effective_isolation(row),
        backend=backend,
        handle=handle,
        runner_image_ref=runner_image_ref,
        runner_image_digest=runner_image_digest,
        sandbox={}
        if runtime.startswith("gym") and row.get("execution_snapshot") is not None
        else _selected_env(_RUNTIME_ENV),
    )


def _task_identifiers(row: Mapping[str, Any]) -> TaskIdentifierProvenance:
    image_ref = _clean_optional(row.get("image_ref"))
    image_digest = _clean_optional(row.get("image_digest") or row.get("task_image_digest"))
    return TaskIdentifierProvenance(
        task_id=str(row["task_id"]),
        task_revision=int(row["task_revision"]),
        task_slug=_clean_optional(row.get("task_slug")),
        image_ref=image_ref,
        image_digest=image_digest,
        image_tag=_image_tag(image_ref),
        tarball_sha256=_clean_optional(row.get("tarball_sha256")),
        tarball_object_key=_clean_optional(row.get("tarball_object_key")),
    )


def _agent_bundle(row: Mapping[str, Any]) -> AgentIdentifierProvenance | None:
    evaluation = snapshot_evaluation(row)
    metadata = _mapping(evaluation.get("runner_metadata")) or _mapping(row.get("runner_metadata"))
    bundle = _mapping(metadata.get("agent_bundle"))
    if not bundle:
        return None
    image_ref = _clean_optional(bundle.get("image_ref"))
    return AgentIdentifierProvenance(
        bundle_id=_clean_optional(bundle.get("id") or bundle.get("bundle_id")),
        bundle_name=_clean_optional(bundle.get("bundle_name")),
        agent_name=_clean_optional(bundle.get("agent_name")),
        agent_version=_clean_optional(bundle.get("agent_version")),
        image_ref=image_ref,
        image_digest=_clean_optional(bundle.get("image_digest")),
        image_tag=_image_tag(image_ref),
        source_lock_digest=_clean_optional(bundle.get("source_lock_digest")),
        fingerprint=_clean_optional(bundle.get("fingerprint")),
        entrypoint=_clean_optional(bundle.get("entrypoint")),
        platform=_clean_optional(bundle.get("platform")),
        runtime_abi=_clean_optional(bundle.get("runtime_abi")),
        bundle_layout_version=_clean_int(bundle.get("bundle_layout_version")),
        builder_profile=_clean_optional(bundle.get("builder_profile")),
    )


def _identifier_summary(
    row: Mapping[str, Any],
    *,
    task_identifiers: TaskIdentifierProvenance,
    agent_bundle: AgentIdentifierProvenance | None,
) -> dict[str, Any]:
    metadata = _mapping(row.get("runner_metadata"))
    qualification = _mapping(metadata.get("qualification"))
    summary: dict[str, Any] = {
        "task": task_identifiers.model_dump(exclude_none=True),
    }
    if agent_bundle is not None:
        summary["agent_bundle"] = agent_bundle.model_dump(exclude_none=True)
    runner = _runner_identifiers(row, metadata=metadata)
    if runner:
        summary["runner"] = runner
    qualification_refs = _qualification_identifiers(qualification)
    if qualification_refs:
        summary["qualification"] = qualification_refs
    control_plane = _control_plane().model_dump(exclude_none=True)
    if control_plane:
        summary["control_plane"] = control_plane
    return summary


def _runner_identifiers(row: Mapping[str, Any], *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = snapshot_evaluation(row)
    metadata = _mapping(evaluation.get("runner_metadata")) or metadata
    artifact = _mapping(metadata.get("artifact"))
    image_ref = _clean_optional(evaluation.get("runner_image_ref")) or _clean_optional(artifact.get("image_ref"))
    identifiers = {
        "image_ref": image_ref,
        "image_digest": _clean_optional(evaluation.get("runner_image_digest"))
        or _clean_optional(artifact.get("image_digest")),
        "image_tag": _image_tag(image_ref),
        "source_revision": _clean_optional(artifact.get("source_revision")),
        "ci_pipeline_id": _clean_optional(artifact.get("ci_pipeline_id")),
        "ci_job_id": _clean_optional(artifact.get("ci_job_id")),
        "framework": _clean_optional(row.get("framework")),
        "requested_framework_version": _clean_optional(row.get("requested_framework_version")),
        "framework_version": _clean_optional(row.get("framework_version")),
        "framework_adapter_version": _clean_optional(row.get("framework_adapter_version")),
        "sandbox_k8s_version": _clean_optional(row.get("sandbox_k8s_version")),
    }
    return {key: value for key, value in identifiers.items() if value is not None}


def _qualification_identifiers(qualification: Mapping[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name in ("release", "adapter", "sandbox_k8s", "validation"):
        value = _mapping(qualification.get(name))
        if value:
            refs[name] = {
                key: item
                for key, item in value.items()
                if item is not None and _identifier_key(key) and not _is_secret_key(str(key))
            }
    return {key: value for key, value in refs.items() if value}


def _harbor(row: Mapping[str, Any]) -> HarborProvenance | None:
    evaluation = snapshot_evaluation(row)
    if str(evaluation.get("framework") or row.get("framework") or "") != "harbor":
        return None
    metadata = _mapping(evaluation.get("runner_metadata")) or _mapping(row.get("runner_metadata"))
    qualification = _mapping(metadata.get("qualification"))
    release = _mapping(qualification.get("release"))
    adapter = _mapping(qualification.get("adapter"))
    sandbox_k8s = _mapping(qualification.get("sandbox_k8s"))
    gym_runtime = str(evaluation.get("runtime") or row.get("runtime") or "").startswith("gym")
    artifact = _mapping(metadata.get("framework_artifact") or metadata.get("artifact"))
    runner_image_ref = _clean_optional(artifact.get("image_ref"))
    if not gym_runtime:
        runner_image_ref = runner_image_ref or _clean_optional(evaluation.get("runner_image_ref"))
    dataset_images = _harbor_dataset_images(row)
    harbor = HarborProvenance(
        profile_id=_clean_optional(row.get("harbor_profile_id")),
        requested_version=_clean_optional(row.get("requested_framework_version")),
        version=_clean_optional(row.get("framework_version")) or _clean_optional(release.get("version")),
        git_commit_sha=_clean_optional(
            release.get("git_commit_sha")
            or release.get("commit_sha")
            or release.get("commit")
            or release.get("revision")
        ),
        wheel_sha256=_clean_optional(release.get("wheel_sha256")),
        adapter_version=_clean_optional(row.get("framework_adapter_version"))
        or _clean_optional(adapter.get("version")),
        adapter_bundle_sha256=_clean_optional(adapter.get("bundle_sha256")),
        sandbox_k8s_version=_clean_optional(row.get("sandbox_k8s_version"))
        or _clean_optional(sandbox_k8s.get("version")),
        sandbox_k8s_git_commit_sha=_clean_optional(
            sandbox_k8s.get("git_commit_sha")
            or sandbox_k8s.get("commit_sha")
            or sandbox_k8s.get("commit")
            or sandbox_k8s.get("revision")
        ),
        runner_image_ref=runner_image_ref,
        runner_image_digest=_clean_optional(artifact.get("image_digest"))
        or (None if gym_runtime else _clean_optional(evaluation.get("runner_image_digest"))),
        runner_image_tag=_image_tag(runner_image_ref),
        runner_source_revision=_clean_optional(artifact.get("source_revision")),
        runner_ci_pipeline_id=_clean_optional(artifact.get("ci_pipeline_id")),
        runner_ci_job_id=_clean_optional(artifact.get("ci_job_id")),
        signature_ref=_clean_optional(artifact.get("signature_ref")),
        signature_digest=_clean_optional(artifact.get("signature_digest")),
        signature_audit_id=_clean_optional(artifact.get("signature_audit_id")),
        profile_config_hash=_optional_value_hash(row.get("harbor_config")),
        dataset_images=dataset_images,
    )
    return harbor if harbor.model_dump(exclude_none=True) else None


def _harbor_dataset_images(row: Mapping[str, Any]) -> list[HarborDatasetImageProvenance]:
    values = backend_handle_raw(row.get("backend_handle")).get("harbor_dataset_image_imports")
    if not isinstance(values, list):
        return []
    images: list[HarborDatasetImageProvenance] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        try:
            images.append(HarborDatasetImageProvenance.model_validate(value))
        except ValueError:
            continue
    return images


def _gym(row: Mapping[str, Any]) -> GymProvenance | None:
    identity = gym_run_identity(row)
    if identity is None:
        return None
    return GymProvenance.model_validate(identity)


def _intake(row: Mapping[str, Any]) -> IntakeProvenance | None:
    profile_id = _clean_optional(row.get("intake_profile_id"))
    config = _mapping(row.get("intake_config"))
    if profile_id is None and not config:
        return None
    endpoint = _clean_optional(config.get("endpoint") or config.get("base_url")) or _first_env("INTAKE_BASE_URL")
    intake = IntakeProvenance(
        profile_id=profile_id,
        endpoint=endpoint,
        workspace=_clean_optional(config.get("workspace")),
        source=_clean_optional(config.get("source")) or _first_env("INTAKE_SOURCE"),
        app=_clean_optional(config.get("app")),
        task=_clean_optional(config.get("task")),
        experiment_id=_clean_optional(config.get("experiment_id")),
        profile_config_hash=_optional_value_hash(config),
    )
    return intake if intake.model_dump(exclude_none=True) else None


def _config_hashes(row: Mapping[str, Any]) -> dict[str, str]:
    hashes = {}
    for key in (
        "harbor_config",
        "framework_config",
        "switchyard_config",
        "intake_config",
    ):
        value = row.get(key)
        if value not in (None, {}, []):
            hashes[key] = _value_hash(value)
    return hashes


def _optional_value_hash(value: Any) -> str | None:
    if value in (None, {}, []):
        return None
    return _value_hash(value)


def _credential_refs(
    credentials: Mapping[str, Any], *, snapshot: Mapping[str, Any] | None
) -> list[CredentialProvenance]:
    refs = []
    snap_credentials = _mapping(snapshot.get("credentials")) if snapshot else {}
    for role, credential_id in sorted(credentials.items()):
        credential_id_text = str(credential_id)
        snap = _mapping(snap_credentials.get(str(role)))
        refs.append(
            CredentialProvenance(
                role=str(role),
                credential_id=credential_id_text,
                fingerprint=str(snap.get("fingerprint") or _short_fingerprint(credential_id_text)),
                provider=_clean_optional(snap.get("provider")),
                payload_kind=_clean_optional(snap.get("payload_kind")),
                fingerprint_source="stored-credential" if snap else "legacy-id-derived",
            )
        )
    return refs


def _switchyard(row: Mapping[str, Any]) -> SwitchyardProvenance | None:
    raw = _switchyard_raw(row)
    if raw is None:
        return None
    endpoint = _clean_optional(raw.get("endpoint"))
    if endpoint is None:
        return None
    config = _mapping(row.get("switchyard_config"))
    return SwitchyardProvenance(
        profile_id=_clean_optional(raw.get("profile_id")),
        topology=(
            row.get("switchyard_topology")
            if row.get("switchyard_topology") in {"shared_campaign", "dedicated", "dedicated_retry"}
            else None
        ),
        mode=str(raw.get("mode") or "managed"),
        namespace=_clean_optional(raw.get("namespace")),
        deployment=_clean_optional(raw.get("name")),
        service=_clean_optional(raw.get("service_name")),
        config_map=_clean_optional(raw.get("config_map_name")),
        secret=_clean_optional(raw.get("secret_name")),
        network_policy=_clean_optional(raw.get("network_policy_name")),
        endpoint=endpoint,
        openai_base_url=_clean_optional(raw.get("openai_base_url")),
        anthropic_base_url=_clean_optional(raw.get("anthropic_base_url")),
        inbound=_clean_optional(raw.get("inbound")),
        book_mode=_clean_optional(raw.get("book_mode") or _switchyard_book_mode(row)),
        port=_clean_int(raw.get("port")),
        manifest_hash=_clean_optional(raw.get("manifest_hash")),
        config_hash=_clean_optional(raw.get("config_hash")),
        drain_until=_timestamp_text(
            row.get("switchyard_drain_until") or _mapping(row.get("switchyard_resource")).get("drain_until")
        ),
        artifact_path=_clean_optional(raw.get("artifact_path") or "switchyard/"),
        endpoint_identity=_clean_optional(raw.get("endpoint_identity")),
        trust_warning=_clean_optional(raw.get("trust_warning")),
        image_ref=_clean_optional(raw.get("image_ref") or config.get("image")),
        image_digest=_clean_optional(raw.get("image_digest") or config.get("image_digest")),
        source_project=_clean_optional(raw.get("source_project") or config.get("source_project")),
        source_ref=_clean_optional(raw.get("source_ref") or config.get("source_ref")),
        source_commit=_clean_optional(raw.get("source_commit") or config.get("source_commit")),
        context_path=_clean_optional(raw.get("context_path") or config.get("context_path")),
        dockerfile_path=_clean_optional(raw.get("dockerfile_path") or config.get("dockerfile_path")),
        dockerfile_sha256=_clean_optional(raw.get("dockerfile_sha256") or config.get("dockerfile_sha256")),
        context_hash=_clean_optional(raw.get("context_hash") or config.get("context_hash")),
    )


def _switchyard_book_mode(row: Mapping[str, Any]) -> str | None:
    config = _mapping(row.get("switchyard_config"))
    value = config.get("book_mode")
    return str(value) if value in {"closed", "open"} else None


def _effective_isolation(row: Mapping[str, Any]) -> EffectiveIsolationProvenance:
    policy = str(row.get("network_policy") or "unrestricted")
    has_switchyard = bool(row.get("switchyard_profile_id"))
    book_mode = _switchyard_book_mode(row)
    direct_egress = {
        "unrestricted": "unrestricted",
        "default_deny": "experiment_resources_only" if has_switchyard else "denied",
        "scoped_egress": "scoped",
    }.get(policy, "unknown")
    model_gateway = {
        "closed": "restricted",
        "open": "open",
    }.get(book_mode, "profile_defined" if has_switchyard else "none")
    bypass_resistant: bool | None = None
    warnings: list[str] = []
    if book_mode == "closed":
        if policy == "default_deny":
            warnings.append(
                "Switchyard enforces closed-book for the configured inference endpoint; "
                "proxy-only sandbox isolation still requires verification that no broader "
                "namespace or platform egress policy also selects the sandbox"
            )
        elif policy == "unrestricted":
            bypass_resistant = False
            warnings.append(
                "Switchyard enforces closed-book for the configured inference endpoint, "
                "but unrestricted direct sandbox egress may allow a separate "
                "endpoint/credential bypass"
            )
        elif policy == "scoped_egress":
            bypass_resistant = False
            warnings.append(
                "Switchyard enforces closed-book for the configured inference endpoint, "
                "but scoped direct egress may allow a separate endpoint/credential bypass "
                "for allowed destinations"
            )
    return EffectiveIsolationProvenance(
        direct_egress=direct_egress,
        model_gateway=model_gateway,
        bypass_resistant=bypass_resistant,
        platform_verified=False,
        warnings=warnings,
    )


def _switchyard_raw(row: Mapping[str, Any]) -> dict[str, Any] | None:
    resource = _mapping(row.get("switchyard_resource"))
    for value in (
        row.get("switchyard"),
        row.get("switchyard_lease"),
        resource.get("metadata"),
    ):
        if value is None:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _env_hashes() -> dict[str, str]:
    hashes = {}
    for key in sorted(set(_CONTROL_PLANE_ENV + _RUNTIME_ENV)):
        value = os.environ.get(key)
        if value and not _is_secret_key(key):
            hashes[key] = _value_hash(value)
    return hashes


def _sbom_refs(row: Mapping[str, Any]) -> list[ProvenanceReference]:
    refs = []
    gym = gym_run_identity(row)
    gym_external = _mapping(gym.get("external_sbom")) if gym else {}
    if gym_external.get("digest"):
        refs.append(
            ProvenanceReference(
                name="GYM_RUNNER_IMAGE_SBOM_DIGEST",
                digest=str(gym_external["digest"]),
            )
        )
    if gym_external.get("ref"):
        refs.append(
            ProvenanceReference(
                name="GYM_RUNNER_IMAGE_SBOM_REF",
                value=str(gym_external["ref"]),
            )
        )
    for key in sorted(os.environ):
        if not (key.endswith("_SBOM_REF") or key.endswith("_SBOM_DIGEST")):
            continue
        if gym is not None and row.get("execution_snapshot") is not None and key.startswith("GYM_"):
            continue
        if _is_secret_key(key):
            continue
        value = os.environ[key]
        refs.append(
            ProvenanceReference(
                name=key,
                value=value if key.endswith("_SBOM_REF") else None,
                digest=value if key.endswith("_SBOM_DIGEST") else None,
            )
        )
    return refs


def _manifest_notes(artifact_root: Path, row: Mapping[str, Any]) -> list[str]:
    notes = [
        "The bundled CycloneDX document is an incomplete run-composition BOM; "
        "package-level image SBOMs remain external build evidence."
    ]
    if not artifact_root.exists():
        notes.append("artifact root did not exist when manifest was generated")
    gym = gym_run_identity(row)
    gym_external = _mapping(gym.get("external_sbom")) if gym else {}
    if not gym_external.get("ref") and not any(key.endswith("_SBOM_REF") for key in os.environ):
        notes.append("No external package-level image SBOM references were declared")
    return notes


def _generated_at(row: Mapping[str, Any]) -> datetime:
    for key in ("finished_at", "updated_at", "created_at"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).astimezone(UTC)
            except ValueError:
                continue
    return datetime.now(tz=UTC)


def _outcome(row: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    outcome = {
        "status": status,
        "status_detail": redact_secret_text(str(row["status_detail"])) if row.get("status_detail") else None,
        "created_at": _timestamp_text(row.get("created_at")),
        "finished_at": _timestamp_text(row.get("finished_at")),
        "reward": row.get("reward"),
        "reward_value": _redacted_value(row.get("reward_value")),
        "n_trials": row.get("n_trials"),
        "n_completed": row.get("n_completed"),
        "n_errored": row.get("n_errored"),
        "n_failed_solve": row.get("n_failed_solve"),
        "exception_counts": _redacted_value(row.get("exception_counts") or {}),
        "result_sha256": canonical_sha256(row["result"]) if row.get("result") is not None else None,
    }
    return {key: value for key, value in outcome.items() if value is not None}


def _selected_env(keys: Sequence[str]) -> dict[str, str]:
    return {key: value for key in keys if (value := os.environ.get(key)) and not _is_secret_key(key)}


def _tool_versions(tools: Sequence[str]) -> dict[str, str]:
    versions = {}
    for tool in tools:
        try:
            completed = subprocess.run(
                [tool, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode == 0 and output:
            versions[tool] = redact_secret_text(output.splitlines()[0])
    return versions


def _file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _value_hash(value: Any) -> str:
    text = json.dumps(_redacted_value(value), sort_keys=True, default=str, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _redacted_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _is_secret_key(str(key)) else _redacted_value(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted_value(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def _short_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value and not _is_secret_key(key):
            return _clean_optional(value)
    return None


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _timestamp_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return _clean_optional(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _image_tag(image_ref: str | None) -> str | None:
    if image_ref is None or "@" in image_ref:
        return None
    _, separator, tag = image_ref.rpartition(":")
    if not separator or "/" in tag:
        return None
    return tag or None


def _identifier_key(key: Any) -> bool:
    normalized = str(key).lower()
    return any(
        marker in normalized for marker in ("sha", "digest", "tag", "version", "revision", "commit", "ref", "id")
    )


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(keyword in normalized for keyword in _SECRET_KEYWORDS)
