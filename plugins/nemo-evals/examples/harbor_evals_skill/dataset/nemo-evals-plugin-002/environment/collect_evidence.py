# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect the inner agent-evaluation job and skill evidence for the separate verifier."""

import errno
import hashlib
import io
import json
import os
import re
import signal
import stat
import sys
import tarfile
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
BUNDLE_LIMIT = 1024 * 1024
TEMPLATE = Path("/opt/helix-task/agent-eval-template.json")
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


def download_results(client: EvaluatorClient, job_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = client.download_agent_eval_job_result(job=job_name, name="agent-eval-results")
    data = bytearray()
    with response.stream(chunk_size=16 * 1024) as chunks:
        for chunk in chunks:
            data.extend(chunk)
            if len(data) > BUNDLE_LIMIT:
                raise ValueError("inner agent-evaluation bundle exceeds its byte limit")

    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in {"summary.json", "trials.jsonl"}:
                continue
            if name in files or not member.isfile() or member.size > JSON_LIMIT:
                raise ValueError(f"invalid result bundle entry {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"missing data for result bundle entry {name}")
            with stream:
                files[name] = stream.read()
    return json_object(files["summary.json"]), [json_object(line) for line in files["trials.jsonl"].splitlines()]


def matches_target(spec: dict[str, Any], expected: dict[str, Any]) -> bool:
    target = spec.get("target")
    if not isinstance(target, dict) or spec.get("trials") is not None or target.get("kind") != expected["kind"]:
        return False
    agent = target.get("agent")
    return isinstance(agent, dict) and all(agent.get(key) == value for key, value in expected["agent"].items())


def collect(workspace_path: Path, evidence_path: Path, staging_path: Path, agent_home: Path, http: httpx.Client) -> int:
    snapshot: dict[str, Any] = {"schema_version": 1, "outcome": "collected"}
    phase = "collecting skill inventory"
    try:
        write_json_atomic(evidence_path / "skill-inventory.json", skill_inventory(staging_path, agent_home))

        phase = "checking inner services"
        for url in (BASE_URL + "/health/ready", "http://127.0.0.1:8091/healthz"):
            ready = http.get(url)
            ready.raise_for_status()
            if ready.json() != {"status": "ready"}:
                raise ValueError("an inner service did not report ready")
        client = EvaluatorClient(
            base_url=BASE_URL,
            workspace=INNER_WORKSPACE,
            http_client=http,
            default_headers={"Accept-Encoding": "identity"},
            timeout=4,
            retry=RetryPolicy(max_retries=0),
        )

        phase = "reading the submitted report"
        with directory(workspace_path) as workspace:
            report = read_agent_json(workspace, "report.json")
        job_name = report.get("job_name")
        if not isinstance(job_name, str) or not re.fullmatch(NAME_PATTERN, job_name):
            raise InvalidSubmission("report.json must include a valid job name")
        snapshot["report"] = report

        phase = "reading the server's agent-evaluation job"
        try:
            job = client.get_agent_eval_job(name=job_name).data()
        except NotFoundError as exc:
            raise InvalidSubmission("reported agent-evaluation job was not found") from exc
        status = client.get_agent_eval_job_status(name=job_name).data()
        if not job.id or job.id != status.id or job.name != job_name or status.name != job_name or job.spec is None:
            raise ValueError("the service returned an inconsistent job record")
        expected = json_object((evidence_path.parent / "root/expected-task.json").read_bytes())
        target = json_object(TEMPLATE.read_bytes())["target"]
        tasks = job.spec["tasks"]
        provenance = [task["spec"].get("provenance") for task in tasks]
        target_matches = matches_target(job.spec, target)
        if status.status.value in {"error", "failed"} and target_matches and provenance == [expected]:
            raise ValueError("the seeded inner agent evaluation failed")
        job_snapshot = {
            "name": job.name,
            "id": job.id,
            "status": status.status.value,
            "run_alias": job.spec.get("labels", {}).get("run_alias"),
            "target_matches": target_matches,
            "task_provenance": provenance,
            "task_ids": [task["id"] for task in tasks],
        }
        snapshot.update(job=job_snapshot, expected_task=expected)
        if status.status.value == "completed":
            phase = "downloading the server's agent-evaluation results"
            job_snapshot["summary"], job_snapshot["trials"] = download_results(client, job_name)
    except InvalidSubmission as exc:
        snapshot.update(outcome="invalid_submission", reason=str(exc))
    except Exception as exc:
        detail = f"HTTP {exc.status_code}" if isinstance(exc, NemoHTTPError) else str(exc)
        snapshot.update(outcome="error", reason=f"{phase}: {type(exc).__name__}: {detail}")

    write_json_atomic(evidence_path / "job-snapshot.json", snapshot)
    if "reason" in snapshot:
        print(snapshot["reason"], file=sys.stderr)
    return int(snapshot["outcome"] == "error")


def collection_timeout(_signum: int, _frame: FrameType | None) -> None:
    raise TimeoutError("collection deadline exceeded")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, collection_timeout)
    # Leave time within Harbor's 30-second hook budget to save the error snapshot.
    signal.alarm(24)
    with httpx.Client(timeout=4, trust_env=False) as http:
        sys.exit(collect(Path("/workspace"), EVIDENCE_DIR, Path("/harbor/skills"), Path("/home/harbor-agent"), http))
