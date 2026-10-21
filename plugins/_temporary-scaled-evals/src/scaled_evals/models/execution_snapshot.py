# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private submission snapshots and their safe public provenance projection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from scaled_evals.api.redaction import redact_secret_text

EXECUTION_SNAPSHOT_SCHEMA_VERSION = "scaled-evals-execution-inputs-v1"
_SECRET_KEYS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "encrypted_payload",
    "byok",
)


def current_process_identity() -> dict[str, str]:
    """Return declared release identity for the process capturing/executing a run."""
    candidates = {
        "package_version": _package_version(),
        "source_revision": _first_env("SCALED_EVALS_SOURCE_REVISION", "CI_COMMIT_SHA", "GIT_COMMIT", "GIT_SHA"),
        "image_ref": _first_env("SCALED_EVALS_IMAGE_REF", "CONTROL_PLANE_IMAGE", "API_IMAGE"),
        "image_digest": _first_env("SCALED_EVALS_IMAGE_DIGEST", "CONTROL_PLANE_IMAGE_DIGEST", "API_IMAGE_DIGEST"),
        "release_version": _first_env("SCALED_EVALS_RELEASE_VERSION", "APP_VERSION"),
        "ci_pipeline_id": _first_env("SCALED_EVALS_CI_PIPELINE_ID", "CI_PIPELINE_ID"),
        "ci_job_id": _first_env("SCALED_EVALS_CI_JOB_ID", "CI_JOB_ID"),
        "signature_ref": _first_env("SCALED_EVALS_SIGNATURE_REF"),
        "signature_digest": _first_env("SCALED_EVALS_SIGNATURE_DIGEST"),
        "signature_audit_id": _first_env("SCALED_EVALS_SIGNATURE_AUDIT_ID"),
    }
    return {key: value for key, value in candidates.items() if value}


def build_execution_snapshot(
    *,
    captured_at: str,
    evaluation: Mapping[str, Any],
    task: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    credentials: Mapping[str, str],
    submission_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact private input snapshot stored with an evaluation.

    Profile config is intentionally exact because dispatch consumes it. Credentials
    are provider -> platform secret references; secret values are never accepted here.
    """
    return {
        "schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
        "captured_at": captured_at,
        "evaluation": _json_value(dict(evaluation)),
        "task": _json_value(dict(task)),
        "profiles": {role: _json_value(dict(profile)) for role, profile in sorted(profiles.items())},
        "credentials": dict(sorted(credentials.items())),
        "submission_identity": _json_value(dict(submission_identity)),
    }


def validate_execution_snapshot(value: Any) -> dict[str, Any] | None:
    """Return a supported snapshot, ``None`` for an explicitly legacy row.

    A malformed or unknown non-null snapshot fails closed rather than silently
    reading mutable profile rows.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("execution snapshot must be a JSON object")
    if value.get("schema_version") != EXECUTION_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing execution snapshot schema_version")
    for key in ("evaluation", "task", "profiles", "credentials", "submission_identity"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"execution snapshot field {key!r} must be an object")
    return value


def snapshot_profile_config(snapshot: Mapping[str, Any], role: str) -> dict[str, Any]:
    profiles = snapshot.get("profiles")
    profile = profiles.get(role) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        return {}
    config = profile.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"snapshotted {role} profile config must be a JSON object")
    return dict(config)


def public_execution_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project a private snapshot into non-secret, hash-bound public evidence."""
    validated = validate_execution_snapshot(dict(snapshot))
    if validated is None:  # pragma: no cover - Mapping input cannot be None
        return {}
    evaluation = validated["evaluation"]
    public_evaluation = {
        key: _sanitize_public(value)
        for key, value in evaluation.items()
        if key not in {"instruction_prefix", "instruction_postfix", "initial_user_turns"}
    }
    for key in ("instruction_prefix", "instruction_postfix", "initial_user_turns"):
        if key in evaluation and evaluation[key] not in (None, [], ""):
            public_evaluation[f"{key}_sha256"] = canonical_sha256(evaluation[key])

    task = validated["task"]
    build_payload = task.get("build_payload")
    public_task = {key: _sanitize_public(value) for key, value in task.items() if key != "build_payload"}
    if build_payload not in (None, {}):
        public_task["build_payload_sha256"] = canonical_sha256(build_payload)
        public_task["build_materials"] = _sanitize_public(build_payload)

    public_profiles: dict[str, Any] = {}
    for role, profile in validated["profiles"].items():
        public_profiles[role] = {
            "id": profile.get("id"),
            "name": profile.get("name"),
            "type": profile.get("type"),
            "updated_at": profile.get("updated_at"),
            "config_sha256": canonical_sha256(profile.get("config") or {}),
        }

    return {
        "schema_version": validated["schema_version"],
        "captured_at": validated.get("captured_at"),
        "evaluation": public_evaluation,
        "task": public_task,
        "profiles": public_profiles,
        "credentials": dict(validated["credentials"]),
        "submission_identity": _sanitize_public(validated["submission_identity"]),
        "private_snapshot_sha256": canonical_sha256(validated),
    }


def canonical_sha256(value: Any) -> str:
    body = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _sanitize_public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_public(item) for key, item in value.items() if not _is_secret_key(str(key))}
    if isinstance(value, list):
        return [_sanitize_public(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return _json_value(value)


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEYS)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _first_env(*names: str) -> str | None:
    return next((value for name in names if (value := os.environ.get(name))), None)


def _package_version() -> str:
    try:
        return importlib.metadata.version("scaled-evals")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
