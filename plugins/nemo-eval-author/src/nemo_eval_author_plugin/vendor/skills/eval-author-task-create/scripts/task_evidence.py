#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain revision-bound native execution evidence using predeclared result checks.

The caller selects the trusted native command and its JSON result contract. This
does not sandbox that command or establish semantic correctness of its checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "nemo.eval_author.task_evidence.v1"
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


class EvidenceError(ValueError):
    """Evidence is missing, stale, or inconsistent."""


def loads(data: str | bytes) -> Any:
    def constant(value: str) -> Any:
        raise EvidenceError(f"nonfinite JSON value: {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvidenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read(path: Path) -> dict[str, Any]:
    try:
        value = loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot read evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"expected regular evidence file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_digest(task: Path) -> str:
    if not task.is_dir() or task.is_symlink():
        raise EvidenceError("task must be a directory, not a symlink")
    entries = []
    for path in sorted(task.rglob("*")):
        relative = path.relative_to(task)
        if any(part in IGNORED for part in relative.parts):
            continue
        if path.is_symlink():
            raise EvidenceError(f"task symlinks are not supported: {relative}")
        entries.append([str(relative), path.stat().st_mode & 0o777, digest(path) if path.is_file() else None])
    if not entries:
        raise EvidenceError("task is empty")
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def pointer(value: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith("/"):
        raise EvidenceError("result checks require an absolute JSON pointer")
    try:
        for token in path[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EvidenceError(f"missing result field: {path}") from exc
    return value


def assertions(value: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    outcomes = []
    for check in checks:
        actual = pointer(value, check["pointer"])
        if "equals" in check:
            outcomes.append(type(actual) is type(check["equals"]) and actual == check["equals"])
        else:
            outcomes.append(
                type(actual) in (int, float) and math.isfinite(actual) and check["min"] <= actual <= check["max"]
            )
    return all(outcomes)


def validate_contract(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise EvidenceError("contract must be an object")
    if not isinstance(contract.get("task_id"), str) or not contract["task_id"].strip():
        raise EvidenceError("contract requires task_id")
    if contract.get("provider") not in ("harbor", "gym"):
        raise EvidenceError("contract provider must be harbor or gym")
    if not isinstance(contract.get("native_run_id"), str) or not contract["native_run_id"].startswith("/"):
        raise EvidenceError("contract requires native_run_id JSON pointer")
    if contract["provider"] == "gym" and contract["native_run_id"] != "/native_run_id":
        raise EvidenceError("Gym contracts use the maintained adapter and /native_run_id")
    cases = contract.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise EvidenceError("contract requires cases")
    kinds = set()
    for case in cases.values():
        if not isinstance(case, dict):
            raise EvidenceError("case must be an object")
        kind = case.get("kind")
        if kind not in ("reference", "incorrect", "alternative", "side_effect", "agent", "diagnostic"):
            raise EvidenceError("unknown case kind")
        kinds.add(kind)
        if not isinstance(case.get("requirement"), str) or not case["requirement"].strip():
            raise EvidenceError("each case must explain its requirement and expected behavior")
        for field in ("health", "checks"):
            checks = case.get(field)
            if not isinstance(checks, list) or not checks:
                raise EvidenceError(f"each case requires nonempty {field} assertions")
            for check in checks:
                if not isinstance(check, dict) or set(check) not in ({"pointer", "equals"}, {"pointer", "min", "max"}):
                    raise EvidenceError("check requires pointer and equals, or pointer and min/max")
                if not isinstance(check["pointer"], str) or not check["pointer"].startswith("/"):
                    raise EvidenceError("check requires an absolute JSON pointer")
                bounds = [check[k] for k in ("min", "max") if k in check]
                if bounds and (
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]
                ):
                    raise EvidenceError("check bounds must be finite and ordered")
    if not {"reference", "incorrect", "agent"} <= kinds:
        raise EvidenceError("contract requires reference, incorrect, and agent cases")
    for kind in ("alternative", "side_effect"):
        if kind not in kinds and not (
            isinstance(contract.get(f"{kind}_not_applicable"), str) and contract[f"{kind}_not_applicable"].strip()
        ):
            raise EvidenceError(f"include {kind} case or explain why it is not applicable")


def fresh(path: Path, task: Path) -> None:
    if path.exists() or path.is_symlink() or path.resolve().is_relative_to(task.resolve()):
        raise EvidenceError(f"output must be new and outside the task tree: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare(
    task: Path, contract_path: Path, output: Path, previous: Path | None = None, reason: str | None = None
) -> dict[str, Any]:
    contract = read(contract_path)
    validate_contract(contract)
    fresh(output, task)
    repairs = []
    if previous:
        old = read(previous)
        if old.get("schema") != SCHEMA or old.get("task_dir") != str(task.resolve()):
            raise EvidenceError("repair must refer to the same task's previous manifest")
        if old.get("contract", {}).get("task_id") != contract["task_id"] or not reason or not reason.strip():
            raise EvidenceError("repair requires the same task ID and a concrete reason")
        repairs = old.get("repairs", [])
        if not isinstance(repairs, list) or len(repairs) >= 3:
            raise EvidenceError("three-repair budget exhausted; retain the unresolved task")
        repairs = [*repairs, {"manifest": str(previous.resolve()), "sha256": digest(previous), "reason": reason}]
    elif reason:
        raise EvidenceError("repair reason requires --previous")
    manifest = {
        "schema": SCHEMA,
        "task_dir": str(task.resolve()),
        "task_sha256": task_digest(task),
        "contract": contract,
        "repairs": repairs,
    }
    if contract["provider"] == "harbor":
        try:
            from harbor.models.task.task import Task
        except ImportError as exc:
            raise EvidenceError("Harbor preparation needs the Harbor Python runtime used for execution") from exc

        manifest["harbor_task_checksum"] = Task(task.resolve()).checksum
    if contract["provider"] == "gym":
        from gym_evidence import snapshot

        manifest["gym_inputs"] = snapshot(task.resolve(), contract)
    write_new(output, manifest)
    return manifest


def manifest_at(path: Path) -> dict[str, Any]:
    manifest = read(path)
    if manifest.get("schema") != SCHEMA:
        raise EvidenceError("unsupported manifest schema")
    validate_contract(manifest.get("contract", {}))
    repairs = manifest.get("repairs")
    if not isinstance(repairs, list) or len(repairs) > 3:
        raise EvidenceError("invalid repair ledger")
    for repair in repairs:
        if digest(Path(repair["manifest"])) != repair["sha256"]:
            raise EvidenceError("superseded manifest changed")
    if task_digest(Path(manifest["task_dir"])) != manifest.get("task_sha256"):
        raise EvidenceError("task revision changed; prepare a new manifest and rerun controls and agents")
    return manifest


def native_identity(native: dict[str, Any], manifest: dict[str, Any]) -> str:
    contract = manifest["contract"]
    identity = pointer(native, contract["native_run_id"])
    if not isinstance(identity, str) or not identity.strip():
        raise EvidenceError("native run identity must be a nonempty string")
    if contract["provider"] == "harbor":
        checksum = manifest.get("harbor_task_checksum")
        if not isinstance(checksum, str) or not checksum or native.get("task_checksum") != checksum:
            raise EvidenceError("native Harbor task checksum differs from prepared task")
        if "exception_info" not in native or native["exception_info"] is not None:
            raise EvidenceError("native Harbor trial has missing health evidence or an exception")
    return identity


def run(
    manifest_path: Path,
    case_id: str,
    run_id: str,
    output: Path,
    result: Path,
    trace: Path | None,
    command: list[str],
    timeout: int = 1800,
    purpose: str = "closure",
    gym_source: Path | None = None,
) -> dict[str, Any]:
    manifest = manifest_at(manifest_path)
    if timeout <= 0:
        raise EvidenceError("timeout must be positive")
    manifest_sha = digest(manifest_path)
    case = manifest["contract"]["cases"].get(case_id)
    if case is None or not run_id.strip() or not command:
        raise EvidenceError("known case, nonempty run ID, and command are required")
    if purpose not in ("baseline", "closure"):
        raise EvidenceError("purpose must be baseline or closure")
    if purpose == "closure" and case["kind"] == "agent" and trace is None:
        raise EvidenceError("agent evidence requires a fresh ATIF trace")
    paths = [output, result] + ([trace] if trace is not None else [])
    if len({p.resolve() for p in paths}) != len(paths):
        raise EvidenceError("receipt, native result, and trace must have distinct paths")
    for path in paths:
        fresh(path, Path(manifest["task_dir"]))
    source = None
    if manifest["contract"]["provider"] == "gym":
        from gym_evidence import source_paths

        if gym_source is None:
            raise EvidenceError("Gym recording requires --gym-source")
        source = read(gym_source)
        for path in source_paths(source):
            if path.exists() or path.is_symlink() or path.resolve().is_relative_to(Path(manifest["task_dir"])):
                raise EvidenceError("Gym native outputs must be fresh and outside the task")
    elif gym_source is not None:
        raise EvidenceError("--gym-source applies only to Gym")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "case": case_id,
        "purpose": purpose,
        "gym_source": source,
        "run_id": run_id,
        "command": command,
        "cwd": str(Path.cwd()),
        "result": str(result.resolve()),
        "trace": str(trace.resolve()) if trace else None,
        "status": "infrastructure_error",
    }
    try:
        completed = subprocess.run(
            command, timeout=timeout, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        receipt["returncode"] = completed.returncode
        if completed.returncode:
            raise EvidenceError("native command failed")
        manifest_at(manifest_path)
        if digest(manifest_path) != manifest_sha:
            raise EvidenceError("manifest changed during execution")
        if source is not None:
            from gym_evidence import build

            write_new(result, build(source, manifest, case, purpose, trace))
        native = read(result)
        receipt["native_run_id"] = native_identity(native, manifest)
        receipt["result_sha256"] = digest(result)
        if trace:
            receipt["trace_sha256"] = digest(trace)
        if not assertions(native, case["health"]):
            raise EvidenceError("native health checks failed")
        receipt["status"] = "passed" if assertions(native, case["checks"]) else "behavior_failed"
        if source is not None:
            receipt["health_status"] = native["health"]["verdict"]
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        receipt["error"] = str(exc)
    write_new(output, receipt)
    return receipt


def verify_receipts(paths: list[Path], task_id: str) -> dict[str, dict[str, Any]]:
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise EvidenceError("distinct execution receipts are required")
    manifests = set()
    seen_cases = set()
    seen_runs = set()
    seen_results = set()
    seen_traces = set()
    seen_native_runs = set()
    agents = {}
    contract: dict[str, Any] = {}
    for path in paths:
        receipt = read(path)
        if receipt.get("schema") != SCHEMA or receipt.get("status") != "passed" or receipt.get("returncode") != 0:
            raise EvidenceError("execution receipt is not passed")
        if receipt.get("purpose") != "closure":
            raise EvidenceError("baseline receipts cannot establish gap closure")
        manifest_path = Path(receipt["manifest"])
        manifest = manifest_at(manifest_path)
        sha = digest(manifest_path)
        if sha != receipt.get("manifest_sha256"):
            raise EvidenceError("manifest digest mismatch")
        manifests.add(sha)
        contract = manifest["contract"]
        if contract["task_id"] != task_id:
            raise EvidenceError("manifest task_id differs from selected task")
        case_id = receipt["case"]
        case = contract["cases"].get(case_id)
        if case is None or case["kind"] == "diagnostic":
            raise EvidenceError("diagnostic or unknown case cannot count as acceptance evidence")
        result = Path(receipt["result"])
        if digest(result) != receipt.get("result_sha256"):
            raise EvidenceError("native result digest mismatch")
        native = read(result)
        if contract["provider"] == "gym":
            from gym_evidence import build

            retained_trace = Path(receipt["trace"]) if receipt.get("trace") else None
            rebuilt = build(receipt["gym_source"], manifest, case, "closure", retained_trace)
            if rebuilt != native:
                raise EvidenceError("Gym native source artifacts changed")
        native_run_id = native_identity(native, manifest)
        if native_run_id != receipt.get("native_run_id") or native_run_id in seen_native_runs:
            raise EvidenceError("native run identity is changed or reused")
        seen_native_runs.add(native_run_id)
        if not assertions(native, case["health"]) or not assertions(native, case["checks"]):
            raise EvidenceError("native evidence does not satisfy declared health and behavior checks")
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip() or run_id in seen_runs:
            raise EvidenceError("execution run IDs must be nonempty and distinct")
        if result.resolve() in seen_results:
            raise EvidenceError("native results must be distinct")
        seen_runs.add(run_id)
        seen_results.add(result.resolve())
        seen_cases.add(case_id)
        if case["kind"] == "agent":
            trace = Path(receipt["trace"])
            if digest(trace) != receipt.get("trace_sha256") or trace.resolve() in seen_traces:
                raise EvidenceError("agent trace is changed or reused")
            seen_traces.add(trace.resolve())
            agents[run_id] = receipt
    if len(manifests) != 1:
        raise EvidenceError("all evidence must use the same manifest revision")
    required = {name for name, case in contract["cases"].items() if case["kind"] not in ("agent", "diagnostic")}
    if not required <= seen_cases or len(agents) < 2:
        raise EvidenceError("every declared control and at least two agent runs are required")
    return agents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--task-dir", type=Path, required=True)
    prep.add_argument("--contract", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--previous", type=Path, help="previous manifest when repairing this task")
    prep.add_argument("--reason", help="concrete repair reason; retain all earlier receipts")
    execute = commands.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--case", required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--result", type=Path, required=True)
    execute.add_argument("--trace", type=Path)
    execute.add_argument("--timeout", type=int, default=1800)
    execute.add_argument("--purpose", choices=("baseline", "closure"), default="closure")
    execute.add_argument("--gym-source", type=Path)
    execute.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            payload = prepare(args.task_dir, args.contract, args.out, args.previous, args.reason)
        else:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            payload = run(
                args.manifest,
                args.case,
                args.run_id,
                args.out,
                args.result,
                args.trace,
                command,
                args.timeout,
                args.purpose,
                args.gym_source,
            )
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("status", "passed") == "passed" else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
