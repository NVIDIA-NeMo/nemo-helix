# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve framework selectors to durable runner metadata at submission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scaled_evals.api.settings import settings
from scaled_evals.harbor_runners import resolve_harbor_runner
from scaled_evals.models.gym_identity import GYM_RUNTIME_LANES

# Runtime name evaluations select to run Harbor trials in OpenSandbox sandboxes.
HARBOR_OPENSANDBOX_RUNTIME = "harbor_opensandbox"
# The only Harbor release the runtime accepts: NemoOpenSandboxEnvironment overrides Harbor internals.
HARBOR_OPENSANDBOX_HARBOR_VERSION = "0.20.0"


@dataclass(frozen=True)
class ResolvedFrameworkRunner:
    requested_version: str | None
    version: str | None
    image_ref: str | None
    image_digest: str | None
    adapter_version: str | None
    sandbox_k8s_version: str | None
    harbor_dir: str | None
    metadata: dict[str, Any]


def resolve_framework_runner(
    framework: str, requested_version: str | None, *, runtime: str | None = None
) -> ResolvedFrameworkRunner:
    if runtime == HARBOR_OPENSANDBOX_RUNTIME and framework != "harbor":
        raise ValueError(f"runtime {runtime!r} requires framework 'harbor'")
    if framework == "harbor":
        if runtime == HARBOR_OPENSANDBOX_RUNTIME:
            runner = resolve_harbor_runner(requested_version or HARBOR_OPENSANDBOX_HARBOR_VERSION)
            if runner.version != HARBOR_OPENSANDBOX_HARBOR_VERSION:
                raise ValueError(
                    f"runtime {runtime!r} supports only Harbor {HARBOR_OPENSANDBOX_HARBOR_VERSION}, "
                    f"not {runner.version}"
                )
        else:
            runner = resolve_harbor_runner(requested_version)
        resolved = ResolvedFrameworkRunner(
            requested_version=requested_version,
            version=runner.version,
            image_ref=settings.harbor_runner_artifact_ref or None,
            image_digest=settings.harbor_runner_artifact_digest or None,
            adapter_version=runner.adapter_version,
            sandbox_k8s_version=runner.sandbox_k8s_version,
            harbor_dir=runner.harbor_dir,
            metadata={
                "qualification": runner.qualification_metadata(),
                "artifact": {
                    "image_ref": settings.harbor_runner_artifact_ref or None,
                    "image_digest": settings.harbor_runner_artifact_digest or None,
                    "source_revision": settings.harbor_runner_source_revision or None,
                    "ci_pipeline_id": settings.harbor_runner_ci_pipeline_id or None,
                    "ci_job_id": settings.harbor_runner_ci_job_id or None,
                    "signature_ref": settings.harbor_runner_signature_ref or None,
                    "signature_digest": settings.harbor_runner_signature_digest or None,
                    "signature_audit_id": settings.harbor_runner_signature_audit_id or None,
                },
            },
        )
    else:
        if requested_version is not None:
            raise ValueError(f"framework_version is not supported for framework {framework!r}; omit it")
        resolved = ResolvedFrameworkRunner(
            requested_version=None,
            version=None,
            image_ref=None,
            image_digest=None,
            adapter_version=None,
            sandbox_k8s_version=None,
            harbor_dir=None,
            metadata={},
        )
    return _with_runtime_runner(resolved, runtime=runtime)


_FULL_GIT_SHA = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_SHA256 = re.compile(r"sha256:[0-9a-fA-F]{64}")


def _with_runtime_runner(resolved: ResolvedFrameworkRunner, *, runtime: str | None) -> ResolvedFrameworkRunner:
    lane = GYM_RUNTIME_LANES.get(runtime or "")
    if lane is None:
        return resolved

    source_revision = _exact_source_revision(settings.gym_source_revision)
    image_ref = _known(settings.gym_runner_image)
    if image_ref is None:
        raise ValueError(f"runtime {runtime!r} requires GYM_RUNNER_IMAGE")
    configured_image_digest = _exact_sha256(
        settings.gym_runner_image_digest,
        "GYM_RUNNER_IMAGE_DIGEST",
    )
    reference_digest = _image_reference_digest(image_ref)
    if configured_image_digest and reference_digest and configured_image_digest != reference_digest:
        raise ValueError("GYM_RUNNER_IMAGE_DIGEST does not match the digest in GYM_RUNNER_IMAGE")
    image_digest = configured_image_digest or reference_digest
    sbom_digest = _exact_sha256(
        settings.gym_runner_image_sbom_digest,
        "GYM_RUNNER_IMAGE_SBOM_DIGEST",
    )
    package_version = _known(settings.gym_package_version)
    if settings.gym_runner_mode == "process":
        if image_digest is None:
            raise ValueError(f"hosted runtime {runtime!r} requires an immutable GYM_RUNNER_IMAGE_DIGEST")
        if source_revision is None:
            raise ValueError(f"hosted runtime {runtime!r} requires GYM_SOURCE_REVISION")
    sbom_ref = _known(settings.gym_runner_image_sbom_ref)
    completeness = "complete" if image_ref and image_digest and source_revision else "incomplete"
    gym = {
        "runtime": runtime,
        **lane,
        "package_version": package_version,
        "source_revision": source_revision,
        "image_ref": image_ref,
        "image_digest": image_digest,
        "identity_completeness": completeness,
        "identity_verification": "declared-unverified",
        "external_sbom": {
            "ref": sbom_ref,
            "digest": sbom_digest,
            "subject_digest": image_digest,
        },
    }
    gym = {
        key: ({nested_key: nested for nested_key, nested in value.items() if nested is not None})
        if isinstance(value, dict)
        else value
        for key, value in gym.items()
        if value is not None
    }
    metadata = dict(resolved.metadata)
    if framework_artifact := metadata.get("artifact"):
        metadata["framework_artifact"] = framework_artifact
    metadata["artifact"] = {
        "kind": "gym-runtime-runner",
        "image_ref": image_ref,
        "image_digest": image_digest,
        "source_revision": source_revision,
        "package_version": package_version,
        "external_sbom": gym["external_sbom"],
    }
    metadata["gym"] = gym
    return ResolvedFrameworkRunner(
        requested_version=resolved.requested_version,
        version=resolved.version,
        image_ref=image_ref,
        image_digest=image_digest,
        adapter_version=resolved.adapter_version,
        sandbox_k8s_version=resolved.sandbox_k8s_version,
        harbor_dir=resolved.harbor_dir,
        metadata=metadata,
    )


def _known(value: str | None) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.lower() == "unknown" else text


def _exact_source_revision(value: str | None) -> str | None:
    revision = _known(value)
    if revision is None:
        return None
    if _FULL_GIT_SHA.fullmatch(revision) is None:
        raise ValueError("GYM_SOURCE_REVISION must be a full 40- or 64-character git SHA")
    return revision.lower()


def _exact_sha256(value: str | None, setting_name: str) -> str | None:
    digest = _known(value)
    if digest is None:
        return None
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{setting_name} must be an exact sha256:<64 hex> digest")
    return digest.lower()


def _image_reference_digest(image_ref: str) -> str | None:
    if "@" not in image_ref:
        return None
    return _exact_sha256(image_ref.rsplit("@", 1)[1], "GYM_RUNNER_IMAGE digest")
