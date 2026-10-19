# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CycloneDX run-composition BOM generation.

This BOM records the immutable materials known to scaled-evals. It deliberately
claims ``incomplete`` composition because package inventories for referenced
container images must be produced and attached by their build pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scaled_evals.models.execution_snapshot import current_process_identity
from scaled_evals.models.gym_identity import (
    backend_handle_raw,
    gym_run_identity,
    is_snapshot_backed,
    snapshot_evaluation,
)

SBOM_FILE_NAME = "scaled-evals-sbom.cdx.json"
SBOM_SPEC_VERSION = "1.6"
_UUID_NAMESPACE = uuid.UUID("80b581dc-29df-4f2a-8641-a591353f8f87")


def build_run_sbom(row: Mapping[str, Any], *, artifact_root: Path) -> dict[str, Any]:
    evaluation_id = str(row["id"])
    root_ref = f"urn:scaled-evals:evaluation:{evaluation_id}"
    components: list[dict[str, Any]] = []

    _append_container(
        components,
        bom_ref=f"urn:scaled-evals:task:{row['task_id']}:{row['task_revision']}",
        name=f"task:{row['task_id']}",
        version=str(row["task_revision"]),
        submitted_ref=_clean(row.get("image_ref")),
        expected_digest=_clean(row.get("image_digest")),
        observed_digest=_clean(row.get("observed_task_image_digest")),
    )
    evaluation = snapshot_evaluation(row)
    gym = gym_run_identity(row)
    runner = _append_container(
        components,
        bom_ref=f"urn:scaled-evals:runner:{evaluation_id}",
        name="nemo-gym-runtime-runner" if gym is not None else f"{evaluation.get('framework') or 'evaluation'}-runner",
        version=_clean(gym.get("package_version")) if gym is not None else _clean(evaluation.get("framework_version")),
        submitted_ref=_clean(evaluation.get("runner_image_ref")),
        expected_digest=_clean(evaluation.get("runner_image_digest")),
        observed_digest=_clean(gym.get("observed_runner_image_digest"))
        if gym is not None
        else _clean(row.get("observed_runner_image_digest")),
        include_incomplete=gym is not None,
    )
    gym_external_properties: list[dict[str, str]] = []
    if gym is not None and runner is not None:
        for name in (
            "runtime",
            "provider",
            "agent_path",
            "source_revision",
            "identity_completeness",
            "identity_verification",
            "observed_source_revision",
            "observed_runner_image_id",
        ):
            if value := _clean(gym.get(name)):
                runner["properties"].append({"name": f"scaled-evals:gym:{name.replace('_', '-')}", "value": value})
        gym_external_properties = _attach_gym_external_sbom(runner, gym)

    runner_metadata = evaluation.get("runner_metadata")
    bundle = runner_metadata.get("agent_bundle") if isinstance(runner_metadata, Mapping) else None
    if isinstance(bundle, Mapping):
        _append_container(
            components,
            bom_ref=f"urn:scaled-evals:agent-bundle:{bundle.get('bundle_id') or evaluation_id}",
            name=str(bundle.get("bundle_name") or bundle.get("agent_name") or "agent-bundle"),
            version=_clean(bundle.get("agent_version")),
            submitted_ref=_clean(bundle.get("image_ref")),
            expected_digest=_clean(bundle.get("image_digest")),
            observed_digest=_clean(bundle.get("observed_image_digest")),
        )

    for index, image in enumerate(_harbor_dataset_images(row)):
        component = _append_container(
            components,
            bom_ref=f"urn:scaled-evals:harbor-dataset-image:{evaluation_id}:{index}",
            name=f"harbor-dataset-image:{image.get('task_id') or index}",
            version=_clean(image.get("task_revision")),
            submitted_ref=_clean(image.get("image_ref")),
            expected_digest=_clean(image.get("image_digest")),
            observed_digest=None,
        )
        if component is not None:
            component["properties"].extend(
                [
                    {
                        "name": "scaled-evals:image:upstream-reference",
                        "value": str(image["source_image"]),
                    },
                    {
                        "name": "scaled-evals:image:upstream-immutable-reference",
                        "value": str(image["source_immutable_image"]),
                    },
                ]
            )

    execution_identity = current_process_identity()
    _append_container(
        components,
        bom_ref=f"urn:scaled-evals:control-plane:{evaluation_id}",
        name="scaled-evals-control-plane",
        version=execution_identity.get("release_version") or execution_identity.get("package_version"),
        submitted_ref=execution_identity.get("image_ref"),
        expected_digest=execution_identity.get("image_digest"),
        observed_digest=None,
    )

    components.extend(_skill_components(artifact_root, evaluation_id=evaluation_id))
    components.extend(_observed_skill_components(row, evaluation_id=evaluation_id))
    external_properties = gym_external_properties + _attach_external_sboms(
        components,
        skip_gym=gym is not None and is_snapshot_backed(row),
    )
    dependencies = [{"ref": root_ref, "dependsOn": [item["bom-ref"] for item in components]}]
    dependencies.extend({"ref": item["bom-ref"], "dependsOn": []} for item in components)

    timestamp = _stable_timestamp(row)
    bom = {
        "$schema": f"https://cyclonedx.org/schema/bom-{SBOM_SPEC_VERSION}.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": SBOM_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid5(_UUID_NAMESPACE, evaluation_id)}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "scaled-evals",
                        "version": execution_identity.get("package_version") or "unknown",
                    }
                ]
            },
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "scaled-evals-run",
                "version": evaluation_id,
            },
            "properties": [
                {"name": "scaled-evals:sbom:kind", "value": "run-composition"},
                {"name": "scaled-evals:sbom:completeness", "value": "incomplete"},
                {
                    "name": "scaled-evals:sbom:package-inventory",
                    "value": "external-build-sboms-required",
                },
            ],
        },
        "components": components,
        "dependencies": dependencies,
        "compositions": [{"aggregate": "incomplete", "assemblies": [root_ref]}],
    }
    if external_properties:
        bom["metadata"]["properties"].extend(external_properties)
    return bom


def write_run_sbom(root: Path | str, row: Mapping[str, Any]) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / SBOM_FILE_NAME
    path.write_text(
        json.dumps(build_run_sbom(row, artifact_root=root_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _harbor_dataset_images(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = backend_handle_raw(row.get("backend_handle")).get("harbor_dataset_image_imports")
    if not isinstance(values, list):
        return []
    required = {
        "source_image",
        "source_immutable_image",
        "task_id",
        "task_revision",
        "image_ref",
        "image_digest",
        "runtime_image",
    }
    return [value for value in values if isinstance(value, Mapping) and required <= value.keys()]


def file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _append_container(
    components: list[dict[str, Any]],
    *,
    bom_ref: str,
    name: str,
    version: str | None,
    submitted_ref: str | None,
    expected_digest: str | None,
    observed_digest: str | None,
    include_incomplete: bool = False,
) -> dict[str, Any] | None:
    if not include_incomplete and not any((submitted_ref, expected_digest, observed_digest)):
        return None
    component: dict[str, Any] = {
        "type": "container",
        "bom-ref": bom_ref,
        "name": name,
        "version": version or "unknown",
        "properties": [
            {
                "name": "scaled-evals:image:identity-status",
                "value": "runtime-observed" if observed_digest else "declared-unverified",
            }
        ],
    }
    if submitted_ref:
        component["properties"].append({"name": "scaled-evals:image:submitted-reference", "value": submitted_ref})
    if expected_digest:
        component["properties"].append({"name": "scaled-evals:image:expected-digest", "value": expected_digest})
    if observed_digest:
        component["properties"].append({"name": "scaled-evals:image:observed-digest", "value": observed_digest})
    digest = observed_digest or expected_digest
    digest_hex = _sha256_hex(digest)
    if digest_hex:
        component["hashes"] = [{"alg": "SHA-256", "content": digest_hex}]
    components.append(component)
    return component


def _skill_components(root: Path, *, evaluation_id: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "skills" not in relative.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append(
            {
                "type": "file",
                "bom-ref": f"urn:scaled-evals:skill:{evaluation_id}:{digest}",
                "name": relative.as_posix(),
                "version": digest[:16],
                "hashes": [{"alg": "SHA-256", "content": digest}],
                "properties": [{"name": "scaled-evals:material:observation", "value": "runtime-staged"}],
            }
        )
    return result


def _observed_skill_components(row: Mapping[str, Any], *, evaluation_id: str) -> list[dict[str, Any]]:
    result = []
    values = row.get("extra_skill_materials")
    if not isinstance(values, list):
        return result
    for index, material in enumerate(values):
        if not isinstance(material, Mapping):
            continue
        digest = _sha256_hex(_clean(material.get("sha256")))
        component: dict[str, Any] = {
            "type": "file",
            "bom-ref": f"urn:scaled-evals:extra-skill:{evaluation_id}:{index}",
            "name": str(material.get("staged_path") or material.get("object_key") or index),
            "version": digest[:16] if digest else "unavailable",
            "properties": [
                {
                    "name": "scaled-evals:material:observation",
                    "value": "runtime-staged" if digest else str(material.get("status") or "unverified"),
                },
                {
                    "name": "scaled-evals:material:object-key",
                    "value": str(material.get("object_key") or ""),
                },
            ],
        }
        if digest:
            component["hashes"] = [{"alg": "SHA-256", "content": digest}]
        result.append(component)
    return result


def _attach_gym_external_sbom(component: dict[str, Any], gym: Mapping[str, Any]) -> list[dict[str, str]]:
    external = gym.get("external_sbom")
    if not isinstance(external, Mapping):
        return []
    ref = _clean(external.get("ref"))
    artifact_digest = _clean(external.get("digest"))
    if not ref and not artifact_digest:
        return []
    subject_digest = _clean(external.get("subject_digest"))
    expected_digest = _clean(gym.get("runner_image_digest"))
    binding = "subject-bound" if ref and subject_digest and expected_digest == subject_digest else "unbound"
    properties = [{"name": "scaled-evals:external-sbom:GYM_RUNNER_IMAGE:binding", "value": binding}]
    if subject_digest:
        properties.append(
            {
                "name": "scaled-evals:external-sbom:GYM_RUNNER_IMAGE:subject-digest",
                "value": subject_digest,
            }
        )
    if artifact_digest:
        properties.append(
            {
                "name": "scaled-evals:external-sbom:GYM_RUNNER_IMAGE:artifact-digest",
                "value": artifact_digest,
            }
        )
    if ref:
        details = "; ".join(
            value
            for value in (
                f"artifact_digest={artifact_digest}" if artifact_digest else "",
                f"subject_digest={subject_digest}" if subject_digest else "",
                f"binding={binding}",
            )
            if value
        )
        component.setdefault("externalReferences", []).append({"type": "bom", "url": ref, "comment": details})
    return properties


def _attach_external_sboms(components: list[dict[str, Any]], *, skip_gym: bool = False) -> list[dict[str, str]]:
    bases = sorted(
        {
            key.removesuffix("_SBOM_REF").removesuffix("_SBOM_DIGEST")
            for key in os.environ
            if key.endswith("_SBOM_REF") or key.endswith("_SBOM_DIGEST")
        }
    )
    properties: list[dict[str, str]] = []
    for base in bases:
        if skip_gym and base.startswith("GYM_"):
            continue
        ref = os.environ.get(f"{base}_SBOM_REF")
        sbom_digest = os.environ.get(f"{base}_SBOM_DIGEST")
        subject_digest = os.environ.get(f"{base}_DIGEST")
        target = _external_sbom_component(components, base)
        binding = "subject-bound" if target is not None and subject_digest else "unbound"
        properties.append({"name": f"scaled-evals:external-sbom:{base}:binding", "value": binding})
        if subject_digest:
            properties.append(
                {
                    "name": f"scaled-evals:external-sbom:{base}:subject-digest",
                    "value": subject_digest,
                }
            )
        if sbom_digest:
            properties.append(
                {
                    "name": f"scaled-evals:external-sbom:{base}:artifact-digest",
                    "value": sbom_digest,
                }
            )
        if target is not None and ref:
            details = "; ".join(
                value
                for value in (
                    f"artifact_digest={sbom_digest}" if sbom_digest else "",
                    f"subject_digest={subject_digest}" if subject_digest else "",
                    f"binding={binding}",
                )
                if value
            )
            target.setdefault("externalReferences", []).append({"type": "bom", "url": ref, "comment": details})
        elif ref:
            properties.append({"name": f"scaled-evals:external-sbom:{base}:ref", "value": ref})
    return properties


def _external_sbom_component(components: list[dict[str, Any]], base: str) -> dict[str, Any] | None:
    normalized = base.upper()
    kind = None
    if any(marker in normalized for marker in ("CONTROL_PLANE", "API", "SCALED_EVALS")):
        kind = "control-plane"
    elif any(marker in normalized for marker in ("RUNNER", "HARBOR", "GYM")):
        kind = "runner"
    elif "TASK" in normalized:
        kind = "task"
    elif "AGENT" in normalized:
        kind = "agent-bundle"
    if kind is None:
        return None
    return next(
        (component for component in components if kind in str(component.get("bom-ref"))),
        None,
    )


def _stable_timestamp(row: Mapping[str, Any]) -> str:
    for key in ("finished_at", "updated_at", "created_at"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if value:
            return str(value)
    snapshot = row.get("execution_snapshot")
    if isinstance(snapshot, Mapping) and snapshot.get("captured_at"):
        return str(snapshot["captured_at"])
    return datetime.now(tz=UTC).isoformat()


def _sha256_hex(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.rsplit("sha256:", 1)[-1]
    if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate):
        return candidate
    return None


def _clean(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
