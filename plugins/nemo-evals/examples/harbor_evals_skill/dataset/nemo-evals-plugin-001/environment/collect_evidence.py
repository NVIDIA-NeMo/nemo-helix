# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the inner service's evaluation record and the files Harbor will transfer."""

import errno
import hashlib
import json
import os
import re
import signal
import stat
import sys
import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.entity_naming import NAME_PATTERN
from nemo_helix_plugin.evals.client import EvaluatorClient

BASE_URL = "http://127.0.0.1:8080"
INNER_WORKSPACE = "default"
EVIDENCE_DIR = Path("/var/lib/helix-task/evidence")
JSON_LIMIT = 256 * 1024
SKILL_FILE_LIMIT = 8 * 1024 * 1024
READ_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class InvalidSubmission(ValueError):
    """The agent's reported job or submitted file could not be accepted."""


class InvalidFile(ValueError):
    """The input is a non-regular or oversized file."""


@contextmanager
def directory(path: Path | str, *, parent: int | None = None) -> Generator[int]:
    fd = os.open(path, READ_DIRECTORY, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


def read_regular_file(parent: int, name: str, limit: int) -> bytes:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENXIO, errno.EISDIR}:
            raise InvalidFile(f"{name} must be a regular file") from exc
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise InvalidFile(f"{name} must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            if info.st_size > limit or len(data := stream.read(limit + 1)) > limit:
                raise InvalidFile(f"{name} exceeds the {limit}-byte limit")
            return data
    finally:
        os.close(fd)


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def json_object(data: bytes | bytearray) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def read_agent_json(workspace: int, name: str) -> dict[str, Any]:
    try:
        return json_object(read_regular_file(workspace, name, JSON_LIMIT))
    except FileNotFoundError as exc:
        raise InvalidSubmission(f"missing {name}") from exc
    except (ValueError, RecursionError) as exc:
        raise InvalidSubmission(f"invalid {name}: {exc}") from exc


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def inventory_packages(root_fd: int) -> dict[str, dict[str, str]]:
    packages = {}
    with os.scandir(root_fd) as entries:
        package_names = sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False))
    for name in package_names:
        files = {}
        with directory(name, parent=root_fd) as package_fd:
            for subpath, _, filenames, folder_fd in os.fwalk(dir_fd=package_fd, follow_symlinks=False):
                for filename in filenames:
                    digest = hashlib.sha256(read_regular_file(folder_fd, filename, SKILL_FILE_LIMIT)).hexdigest()
                    files[(Path(subpath) / filename).as_posix()] = digest
        packages[name] = files
    return packages


def skill_inventory(staging_path: Path, agent_home: Path) -> dict[str, dict[str, dict[str, str]]]:
    with directory(staging_path) as staged_fd:
        staged = inventory_packages(staged_fd)
    with ExitStack() as stack:
        home_fd = stack.enter_context(directory(agent_home))
        try:
            agents_fd = stack.enter_context(directory(".agents", parent=home_fd))
            registered_fd = stack.enter_context(directory("skills", parent=agents_fd))
        except FileNotFoundError:
            registered = {}
        else:
            registered = inventory_packages(registered_fd)
    return {"staged": staged, "registered": registered}


def copy_agent_outputs(workspace_path: Path, evidence_path: Path) -> str:
    with directory(workspace_path) as workspace:
        try:
            scores = read_agent_json(workspace, "aggregate-scores.json")
        except InvalidSubmission as exc:
            # The verifier can still grade a real job with a missing or malformed submitted artifact.
            print(exc, file=sys.stderr)
        else:
            write_json_atomic(evidence_path / "agent-aggregate-scores.json", scores)

        report = read_agent_json(workspace, "report.json")
        job_name = report.get("job_name")
        if (
            not isinstance(job_name, str)
            or not re.fullmatch(NAME_PATTERN, job_name)
            or not isinstance(report.get("status"), str)
            or not isinstance(report.get("metric"), str)
            or type(report.get("mean")) not in (int, float)
        ):
            raise InvalidSubmission("report.json must specify a job name, status, metric, and numeric mean")
        write_json_atomic(evidence_path / "agent-report.json", report)
        return job_name


def grading_spec(spec: dict[str, Any]) -> dict[str, Any]:
    metrics = []
    for bundle in spec["metrics"]:
        payload = bundle["payload"]
        metric = payload["metric"] if payload["kind"] == "inline" else {}
        metrics.append(
            {
                "metric_type": bundle["metric_type"],
                "kind": payload["kind"],
                "type": metric.get("type"),
                "reference": metric.get("reference"),
                "candidate": metric.get("candidate"),
            }
        )
    return {
        "dataset": spec["dataset"],
        "metrics": metrics,
        "has_target": spec.get("target") is not None,
        "field_mapping": spec.get("field_mapping"),
        "prompt_template": spec.get("prompt_template"),
    }


def download_aggregate(client: EvaluatorClient, job_name: str) -> dict[str, Any]:
    response = client.download_evaluate_job_aggregate_scores(name=job_name)
    data = bytearray()
    with response.stream(chunk_size=16 * 1024) as chunks:
        for chunk in chunks:
            data.extend(chunk)
            if len(data) > JSON_LIMIT:
                raise ValueError("server aggregate exceeds the JSON byte limit")
    aggregate = json_object(data)
    scores = aggregate.get("scores")
    if not isinstance(scores, list) or any(
        not isinstance(score, dict) or not isinstance(score.get("name"), str) for score in scores
    ):
        raise ValueError("invalid aggregate returned by inner service")
    return aggregate


def collect(workspace_path: Path, evidence_path: Path, staging_path: Path, agent_home: Path, http: httpx.Client) -> int:
    snapshot_path = evidence_path / "job-snapshot.json"
    snapshot: dict[str, Any] = {"schema_version": 1, "outcome": "collected", "job": None}
    reason = None
    phase = "clearing previous evidence"
    try:
        for name in ("job-snapshot.json", "agent-report.json", "agent-aggregate-scores.json", "skill-inventory.json"):
            (evidence_path / name).unlink(missing_ok=True)

        phase = "collecting skill inventory"
        write_json_atomic(evidence_path / "skill-inventory.json", skill_inventory(staging_path, agent_home))

        phase = "checking inner service readiness"
        ready = http.get(f"{BASE_URL}/health/ready")
        ready.raise_for_status()
        if ready.json() != {"status": "ready"}:
            raise ValueError("inner service did not report ready")

        client = EvaluatorClient(
            base_url=BASE_URL,
            workspace=INNER_WORKSPACE,
            http_client=http,
            default_headers={"Accept-Encoding": "identity"},
            timeout=4,
            retry=RetryPolicy(max_retries=0),
        )
        if client.get_health().data().status != "ok":
            raise ValueError("inner Evals service did not report ok")

        phase = "collecting agent outputs"
        job_name = copy_agent_outputs(workspace_path, evidence_path)

        phase = "reading evaluation job"
        try:
            job = client.get_evaluate_job(name=job_name).data()
        except NotFoundError as exc:
            raise InvalidSubmission("reported evaluation job was not found in the inner workspace") from exc
        status = client.get_evaluate_job_status(name=job_name).data()
        if job.name != job_name or status.name != job_name or job.id != status.id or job.spec is None:
            raise ValueError("inner service returned an inconsistent job identity or missing spec")
        job_evidence = {
            "name": job.name,
            "id": job.id,
            "workspace": INNER_WORKSPACE,
            "status": status.status.value,
            "spec": grading_spec(job.spec),
            "aggregate": None,
        }
        snapshot["job"] = job_evidence
        if status.status.value == "completed":
            phase = "downloading server aggregate"
            job_evidence["aggregate"] = download_aggregate(client, job_name)
    except InvalidSubmission as exc:
        snapshot["outcome"] = "invalid_submission"
        reason = str(exc)
    except Exception as exc:
        detail = f"HTTP {exc.status_code}" if isinstance(exc, NemoHTTPError) else str(exc)
        snapshot["outcome"] = "error"
        reason = f"{phase}: {type(exc).__name__}: {detail}"

    snapshot["reason"] = reason
    write_json_atomic(snapshot_path, snapshot)
    if reason is not None:
        print(reason, file=sys.stderr)
    return int(snapshot["outcome"] == "error")


def collection_timeout(_signum: int, _frame: FrameType | None) -> None:
    raise TimeoutError("collection deadline exceeded")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, collection_timeout)
    # Harbor's 30-second hook budget allows time to export a timeout snapshot.
    signal.alarm(24)
    with httpx.Client(timeout=4, trust_env=False) as http:
        sys.exit(collect(Path("/workspace"), EVIDENCE_DIR, Path("/harbor/skills"), Path("/home/harbor-agent"), http))
