#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure one ATIF trace against an audit-spec denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import quote

try:
    from harbor.models.trajectories import Trajectory  # ty: ignore[unresolved-import]
except ImportError as exc:
    HARBOR_IMPORT_ERROR: ImportError | None = exc
    Trajectory = Any  # type: ignore[assignment,misc]
else:
    HARBOR_IMPORT_ERROR = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _markdown import AuditMarkdownError  # noqa: E402
from _schema import AuditEnvironmentError, AuditSpecError, item_counts, load_audit_spec  # noqa: E402
from measurements import capabilities, failure_cases, tool_calls  # noqa: E402

JsonObject: TypeAlias = dict[str, Any]

COVERAGE_SCHEMA = "nemo.eval_author.audit_coverage.v1"
SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
COVERAGE_SCHEMA_PATH = SCHEMAS_DIR / "audit_coverage.schema.json"
CAPABILITY_JUDGMENTS_SCHEMA_PATH = SCHEMAS_DIR / "audit_capability_judgments.schema.json"
FAILURE_CASE_JUDGMENTS_SCHEMA_PATH = SCHEMAS_DIR / "audit_failure_case_judgments.schema.json"
DETAIL_SCHEMA_PATHS = {
    capabilities.DETAILS_SCHEMA: SCHEMAS_DIR / "audit_capabilities_details.schema.json",
    failure_cases.DETAILS_SCHEMA: SCHEMAS_DIR / "audit_failure_cases_details.schema.json",
    tool_calls.DETAILS_SCHEMA: SCHEMAS_DIR / "audit_tool_calls_details.schema.json",
}


@dataclass(frozen=True)
class MeasurementInputs:
    """Optional sidecar inputs shared by selected measurement methods."""

    capability_judgments: JsonObject | None
    failure_case_judgments: JsonObject | None


@dataclass(frozen=True)
class MeasurementMethod:
    """Callable contract and schema metadata for one measurement method."""

    name: str
    details_schema: str
    measure: Callable[[JsonObject, Trajectory, MeasurementInputs], JsonObject]


def _measure_capabilities(audit: JsonObject, trajectory: Trajectory, inputs: MeasurementInputs) -> JsonObject:
    """Run capability coverage with optional skill-authored judgments."""
    return capabilities.measure(audit, trajectory, judgments=inputs.capability_judgments)


def _measure_failure_cases(audit: JsonObject, trajectory: Trajectory, inputs: MeasurementInputs) -> JsonObject:
    """Run failure-case coverage with optional skill-authored judgments."""
    return failure_cases.measure(audit, trajectory, judgments=inputs.failure_case_judgments)


def _measure_tool_calls(audit: JsonObject, trajectory: Trajectory, _inputs: MeasurementInputs) -> JsonObject:
    """Run tool-call coverage without sidecar inputs."""
    return tool_calls.measure(audit, trajectory)


METHODS: dict[str, MeasurementMethod] = {
    capabilities.METHOD_NAME: MeasurementMethod(
        name=capabilities.METHOD_NAME,
        details_schema=capabilities.DETAILS_SCHEMA,
        measure=_measure_capabilities,
    ),
    failure_cases.METHOD_NAME: MeasurementMethod(
        name=failure_cases.METHOD_NAME,
        details_schema=failure_cases.DETAILS_SCHEMA,
        measure=_measure_failure_cases,
    ),
    tool_calls.METHOD_NAME: MeasurementMethod(
        name=tool_calls.METHOD_NAME,
        details_schema=tool_calls.DETAILS_SCHEMA,
        measure=_measure_tool_calls,
    ),
}


class AuditTraceError(ValueError):
    """Raised when the trace input cannot be interpreted."""


class AuditMeasurementError(ValueError):
    """Raised when measurement configuration or output is invalid."""


@dataclass(frozen=True)
class PendingSubject:
    """The trace and caller/provider identity available before parsing ATIF."""

    trace_path: Path
    task_id: str
    run_id: str | None


@dataclass(frozen=True)
class Subject:
    """The trace subject being measured."""

    trace_path: Path
    task_id: str
    run_id: str

    def to_json(self) -> JsonObject:
        return {
            "trace": str(self.trace_path),
            "trace_format": "atif",
            "task_id": self.task_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class LoadedTrace:
    """Parsed ATIF plus a stable fallback identity for ad-hoc traces."""

    trajectory: Trajectory
    content_sha256: str


@dataclass(frozen=True)
class MeasurementReport:
    """In-memory coverage/details output for one measurement method."""

    method_name: str
    coverage: JsonObject
    details: JsonObject


@dataclass(frozen=True)
class WrittenMeasurement:
    """Filesystem paths and aggregation facts written for one method."""

    method_name: str
    item_kind: str
    coverage_path: Path
    details_path: Path
    covered: list[str]

    def to_summary(self) -> JsonObject:
        return {
            "method": self.method_name,
            "item_kind": self.item_kind,
            "coverage": str(self.coverage_path),
            "details": str(self.details_path),
            "covered": self.covered,
            "covered_count": len(self.covered),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True, help="audit.md file to measure against")
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument("--trace", type=Path, help="ATIF trajectory JSON file")
    subject.add_argument("--trial-dir", type=Path, help="Harbor trial directory containing agent/trajectory.json")
    parser.add_argument("--task-id", help="task id to stamp on the measurement report")
    parser.add_argument("--run-id", help="run id to stamp on the measurement report")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory where task=<id>/run=<id>/<method>/coverage.json and details.json will be written",
    )
    parser.add_argument(
        "--measure",
        action="append",
        help=f"comma-separated measurement methods to run (default: {tool_calls.METHOD_NAME})",
    )
    parser.add_argument(
        "--capability-judgments",
        type=Path,
        help="JSON file of skill-authored judgments for non-tool capability evidence",
    )
    parser.add_argument(
        "--failure-case-judgments",
        type=Path,
        help="JSON file of skill-authored judgments for non-tool failure-case evidence",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    try:
        method_names = _measurement_methods(args.measure)
        inputs = _measurement_inputs(args, method_names)
        audit = load_audit_spec(args.audit)
        pending_subject = _subject(args)
        loaded_trace = _load_harbor_trajectory(pending_subject.trace_path)
        _validate_judgment_trace(inputs.capability_judgments, loaded_trace, label="capability judgments")
        _validate_judgment_trace(inputs.failure_case_judgments, loaded_trace, label="failure-case judgments")
        subject_info = _finalize_subject(pending_subject, loaded_trace)
        reports = _measure_all(
            audit=audit,
            audit_path=args.audit,
            trajectory=loaded_trace.trajectory,
            subject=subject_info,
            method_names=method_names,
            inputs=inputs,
        )
        _validate_reports(reports)
        written_reports = _write_reports(reports, out_dir=args.out_dir, subject=subject_info, compact=args.compact)
    except AuditEnvironmentError as exc:
        _print({"valid": None, "written": False, "error_type": "environment", "error": str(exc)}, compact=args.compact)
        return 2
    except (AuditMarkdownError, AuditSpecError) as exc:
        _print({"valid": False, "written": False, "error_type": "audit_spec", "error": str(exc)}, compact=args.compact)
        return 1
    except AuditTraceError as exc:
        _print({"valid": True, "written": False, "error_type": "trace", "error": str(exc)}, compact=args.compact)
        return 1
    except AuditMeasurementError as exc:
        _print({"valid": True, "written": False, "error_type": "measurement", "error": str(exc)}, compact=args.compact)
        return 1

    summary: JsonObject = {
        "valid": True,
        "written": True,
        "audit": str(args.audit),
        "trace": str(subject_info.trace_path),
        "task_id": subject_info.task_id,
        "methods": [report.method_name for report in written_reports],
        "measurements": [report.to_summary() for report in written_reports],
        "run_id": subject_info.run_id,
    }
    _print(summary, compact=args.compact)
    return 0


def _subject(args: argparse.Namespace) -> PendingSubject:
    """Resolve the requested trace and provider-neutral subject identifiers before parsing ATIF."""
    if args.trial_dir is not None:
        return _subject_from_trial_dir(args.trial_dir, task_id=args.task_id, run_id=args.run_id)
    if args.trace is None:
        raise AuditTraceError("provide --trace or --trial-dir")
    return _subject_from_trace(args.trace, task_id=args.task_id, run_id=args.run_id)


def _subject_from_trial_dir(trial_dir: Path, *, task_id: str | None, run_id: str | None) -> PendingSubject:
    """Derive task/run identity from Harbor metadata when a trial directory is supplied."""
    if not trial_dir.is_dir():
        raise AuditTraceError(f"Harbor trial directory does not exist: {trial_dir}")
    trace_path = trial_dir / "agent" / "trajectory.json"
    if not trace_path.exists():
        raise AuditTraceError(
            f"Harbor trial did not emit an ATIF trace at {trace_path}; agents without SUPPORTS_ATIF may omit it"
        )
    result_path = trial_dir / "result.json"
    result = _load_harbor_result(result_path)
    return PendingSubject(
        trace_path=trace_path,
        task_id=task_id or _string(result.get("task_name")) or trial_dir.name,
        run_id=run_id or _string(result.get("trial_name")) or trial_dir.name,
    )


def _subject_from_trace(trace_path: Path, *, task_id: str | None, run_id: str | None) -> PendingSubject:
    """Derive task/run identity from a direct trace path, honoring explicit CLI stamps first."""
    trial_dir = _harbor_trial_dir_for_trace(trace_path)
    result_path = trial_dir / "result.json" if trial_dir is not None else None
    result = _load_harbor_result(result_path) if result_path is not None else {}
    return PendingSubject(
        trace_path=trace_path,
        task_id=task_id or _string(result.get("task_name")) or trace_path.stem,
        run_id=run_id or _string(result.get("trial_name")) or (trial_dir.name if trial_dir is not None else None),
    )


def _harbor_trial_dir_for_trace(trace_path: Path) -> Path | None:
    """Recognize the common Harbor trial layout without putting Harbor fields in report schemas."""
    if trace_path.name == "trajectory.json" and trace_path.parent.name == "agent":
        return trace_path.parent.parent
    return None


def _load_harbor_result(path: Path) -> JsonObject:
    """Read optional Harbor result metadata only to infer provider-neutral task/run ids."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditTraceError(f"could not read Harbor result metadata at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditTraceError(f"Harbor result metadata at {path} must be a JSON object")
    return payload


def _load_harbor_trajectory(path: Path) -> LoadedTrace:
    """Read ATIF through Harbor once; downstream methods receive the typed trajectory object."""
    if HARBOR_IMPORT_ERROR is not None:
        raise AuditEnvironmentError(
            "Harbor is required to read ATIF trajectories for audit measurement; "
            "install requirements.txt or run with uv --with-requirements"
        ) from HARBOR_IMPORT_ERROR
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuditTraceError(f"could not read Harbor ATIF trajectory at {path}: {exc}") from exc
    try:
        trajectory_model: Any = Trajectory
        trajectory = trajectory_model.model_validate_json(raw)
    except ValueError as exc:
        raise AuditTraceError(f"{path} is not an ATIF trajectory accepted by Harbor: {exc}") from exc
    return LoadedTrace(trajectory=trajectory, content_sha256=hashlib.sha256(raw).hexdigest())


def _finalize_subject(subject: PendingSubject, loaded_trace: LoadedTrace) -> Subject:
    """Fill any missing run identity from the parsed trace or trace content."""
    return Subject(
        trace_path=subject.trace_path,
        task_id=subject.task_id,
        run_id=subject.run_id or _trace_run_id(loaded_trace),
    )


def _trace_run_id(loaded_trace: LoadedTrace) -> str:
    """Prefer explicit ATIF identity, then fall back to a short trace-content digest."""
    trajectory = loaded_trace.trajectory
    return (
        _string(getattr(trajectory, "trajectory_id", None))
        or _string(getattr(trajectory, "session_id", None))
        or f"trace-sha256-{loaded_trace.content_sha256[:12]}"
    )


def _measurement_methods(measure_values: list[str] | None) -> list[str]:
    """Parse comma-separated method selections and de-duplicate them in request order."""
    values = measure_values or []
    if not values:
        values = [tool_calls.METHOD_NAME]

    method_names: list[str] = []
    for value in values:
        method_names.extend(name.strip() for name in value.split(",") if name.strip())
    if not method_names:
        raise AuditMeasurementError("--measure must name at least one measurement method")

    unknown = sorted({name for name in method_names if name not in METHODS})
    if unknown:
        available = ", ".join(sorted(METHODS))
        raise AuditMeasurementError(f"unknown measurement method(s): {', '.join(unknown)}; available: {available}")

    return list(dict.fromkeys(method_names))


def _measurement_inputs(args: argparse.Namespace, method_names: list[str]) -> MeasurementInputs:
    """Load optional sidecar inputs and ensure they apply to the selected methods."""
    if args.capability_judgments is not None and capabilities.METHOD_NAME not in method_names:
        raise AuditMeasurementError("--capability-judgments requires --measure capabilities")
    if args.failure_case_judgments is not None and failure_cases.METHOD_NAME not in method_names:
        raise AuditMeasurementError("--failure-case-judgments requires --measure failure_cases")
    return MeasurementInputs(
        capability_judgments=_load_judgments(
            args.capability_judgments,
            schema_path=CAPABILITY_JUDGMENTS_SCHEMA_PATH,
            label="capability judgments",
        ),
        failure_case_judgments=_load_judgments(
            args.failure_case_judgments,
            schema_path=FAILURE_CASE_JUDGMENTS_SCHEMA_PATH,
            label="failure-case judgments",
        ),
    )


def _load_judgments(path: Path | None, *, schema_path: Path, label: str) -> JsonObject | None:
    """Read and validate an optional skill-authored judgment sidecar."""
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditMeasurementError(f"could not read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditMeasurementError(f"{label} at {path} must be a JSON object")
    _validate_report(
        payload,
        schema_path=schema_path,
        label=label,
        generated=False,
    )
    return payload


def _validate_judgment_trace(
    judgments: JsonObject | None,
    loaded_trace: LoadedTrace,
    *,
    label: str,
) -> None:
    """Reject a judgment sidecar authored for different ATIF content."""
    if judgments is None:
        return
    expected = f"sha256:{loaded_trace.content_sha256}"
    actual = judgments["trace_sha256"]
    if actual != expected:
        raise AuditMeasurementError(f"{label} trace_sha256 {actual!r} does not match measured trace {expected!r}")


def _measure_all(
    *,
    audit: JsonObject,
    audit_path: Path,
    trajectory: Trajectory,
    subject: Subject,
    method_names: list[str],
    inputs: MeasurementInputs,
) -> list[MeasurementReport]:
    """Fan the shared trajectory into each selected measurement method."""
    return [
        _measure(
            audit=audit,
            audit_path=audit_path,
            trajectory=trajectory,
            subject=subject,
            method_name=method_name,
            inputs=inputs,
        )
        for method_name in method_names
    ]


def _measure(
    *,
    audit: JsonObject,
    audit_path: Path,
    trajectory: Trajectory,
    subject: Subject,
    method_name: str,
    inputs: MeasurementInputs,
) -> MeasurementReport:
    """Wrap one method's raw result in the shared coverage/details envelope."""
    method = METHODS[method_name]
    try:
        measurement = method.measure(audit, trajectory, inputs)
    except ValueError as exc:
        raise AuditMeasurementError(str(exc)) from exc
    audit_info = _audit_info(audit, audit_path)
    subject_info = subject.to_json()
    method_info = {"name": method.name}
    item_kind = measurement["item_kind"]
    kind_counts = item_counts(audit)
    if not isinstance(item_kind, str) or item_kind not in kind_counts:
        raise AuditMeasurementError(f"measurement method {method.name!r} returned unsupported item kind {item_kind!r}")
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "audit": audit_info,
        "subject": subject_info,
        "method": method_info,
        "item_kind": item_kind,
        "item_kind_count": kind_counts[item_kind],
        "covered": measurement["covered"],
    }
    details = {
        **measurement["details"],
        "audit": audit_info,
        "subject": subject_info,
        "method": method_info,
    }
    return MeasurementReport(method_name=method.name, coverage=coverage, details=details)


def _validate_reports(reports: list[MeasurementReport]) -> None:
    """Validate every generated report before any files are written."""
    for report in reports:
        _validate_report(report.coverage, schema_path=COVERAGE_SCHEMA_PATH, label=f"{report.method_name} coverage")
        _validate_report(
            report.details,
            schema_path=_details_schema_path(report.details),
            label=f"{report.method_name} details",
        )


def _write_reports(
    reports: list[MeasurementReport],
    *,
    out_dir: Path,
    subject: Subject,
    compact: bool,
) -> list[WrittenMeasurement]:
    """Write one coverage/details file pair per selected measurement method."""
    written_reports: list[WrittenMeasurement] = []
    for report in reports:
        method_dir = (
            out_dir
            / _path_component("task", subject.task_id)
            / _path_component("run", subject.run_id)
            / report.method_name
        )
        coverage_path = method_dir / "coverage.json"
        details_path = method_dir / "details.json"
        try:
            method_dir.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(_json(report.coverage, compact=compact) + "\n", encoding="utf-8")
            details_path.write_text(_json(report.details, compact=compact) + "\n", encoding="utf-8")
        except OSError as exc:
            raise AuditEnvironmentError(
                f"could not write {report.method_name} measurement reports under {method_dir}: {exc}"
            ) from exc
        written_reports.append(
            WrittenMeasurement(
                method_name=report.method_name,
                item_kind=report.coverage["item_kind"],
                coverage_path=coverage_path,
                details_path=details_path,
                covered=report.coverage["covered"],
            )
        )
    return written_reports


def _audit_info(audit: JsonObject, audit_path: Path) -> JsonObject:
    """Record only denominator identity, not the audit items themselves."""
    return {
        "path": str(audit_path),
        "schema": audit["schema"],
        "agent": audit["agent"],
        "status": audit["status"],
        "item_count": len(audit["items"]),
    }


def _path_component(label: str, value: str) -> str:
    """Encode externally supplied ids as single safe path components."""
    return f"{label}={quote(value, safe='')}"


def _details_schema_path(details: JsonObject) -> Path:
    """Resolve the JSON Schema for a method-specific details payload."""
    schema = details.get("schema")
    if not isinstance(schema, str) or schema not in DETAIL_SCHEMA_PATHS:
        raise AuditMeasurementError(f"no details JSON Schema is registered for {schema!r}")
    return DETAIL_SCHEMA_PATHS[schema]


def _validate_report(report: JsonObject, *, schema_path: Path, label: str, generated: bool = True) -> None:
    """Validate JSON against a schema committed with the skill."""
    try:
        from jsonschema import Draft202012Validator  # ty: ignore[unresolved-import]
        from jsonschema.exceptions import SchemaError  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise AuditEnvironmentError(f"jsonschema is required to validate {label} reports") from exc

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditEnvironmentError(f"could not load {label} JSON Schema from {schema_path}: {exc}") from exc

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise AuditEnvironmentError(f"bundled {label} JSON Schema is invalid: {exc.message}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        prefix = f"generated {label} report" if generated else label
        raise AuditMeasurementError(
            f"{prefix} failed its JSON Schema: "
            + "\n".join(f"{list(error.absolute_path)}: {error.message}" for error in errors)
        )


def _string(value: object) -> str | None:
    """Return stripped non-empty strings and drop every other value."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _json(payload: JsonObject, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps(payload, sort_keys=True, indent=2)


def _print(payload: JsonObject, *, compact: bool) -> None:
    print(_json(payload, compact=compact))


if __name__ == "__main__":
    raise SystemExit(main())
