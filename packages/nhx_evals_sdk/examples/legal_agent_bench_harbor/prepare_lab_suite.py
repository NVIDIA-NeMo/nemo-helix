# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Harbor task suite for Harvey Labs' LAB — self-contained.

LAB's public repo ships **raw** tasks (`tasks/**/task.json` = title + instructions + rubric criteria,
plus a `documents/` dir); it does not ship ready Harbor tasks. This script downloads the pinned source
(verifying its SHA-256) and *generates* one Harbor task folder per LAB task, producing a plain Harbor
suite that the SDK's `discover_harbor_tasks` / `HarborAgentTaskRunner` consume directly.

Each generated task folder:

    <task_id>/
      task.toml               # Harbor task config (+ optional [verifier.env] judge creds)
      instruction.md          # the LAB instructions the agent is prompted with
      documents/              # the LAB input documents (copied from source)
      environment/Dockerfile  # doc-tooling image (libreoffice/pandoc + extraction libs + openai)
      tests/task.json         # title + criteria (read by the verifier)
      tests/lab_verify.py      # the in-container rubric verifier (copied from this example)
      tests/test.sh           # runs lab_verify.py after the agent

Point `run_legal_agent_bench.py --dataset-path <out-dir>` at the result.

SEAMS TO RECONCILE:
* `tests/test.sh` reads the agent's deliverables from `/logs/agent/artifacts/lab-run` — LAB's reference
  agent's output location. If you run a different Harbor agent, set `--run-dir` to where it writes.
* The verifier's judge prompt is faithful-in-shape, not LAB's verbatim `rubric_criterion` prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

PACKAGE_DIR = Path(__file__).resolve().parent
LAB_VERIFY_SOURCE = PACKAGE_DIR / "lab_verify.py"

LAB_REVISION = "f46ef86e4788545622db25dcffa3aebb7a139929"
LAB_ARCHIVE_URL = f"https://codeload.github.com/harveyai/harvey-labs/tar.gz/{LAB_REVISION}"
LAB_ARCHIVE_SHA256 = "e45cbdf3236b22866e034bcc62fb23bf00ef2f2e49db7a0cd8a4b07dbae9212c"
LAB_ARCHIVE_ROOT = f"harvey-labs-{LAB_REVISION}"
EXPECTED_TASK_COUNT = 1_749

# Doc-tooling image: enough for the verifier to extract text (and for a doc-capable agent to work).
_DOCKERFILE = """FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        bash ca-certificates curl libreoffice pandoc poppler-utils ripgrep \\
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \\
    && python -m pip install \\
        "openai>=1.50.0" "markitdown>=0.1.0" "openpyxl>=3.1.0" "pandas>=2.0.0" \\
        "pdfplumber>=0.10.0" "python-docx>=1.1.0" "python-pptx>=0.6.23"

WORKDIR /workspace/output
"""

# Runs the in-container verifier after the agent. Mirrors LAB's reference-agent output location; see
# the module docstring's "SEAMS TO RECONCILE".
_TEST_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
python /tests/lab_verify.py \\
  --task-json /tests/task.json \\
  --run-dir {run_dir} \\
  --verifier-dir /logs/verifier \\
  --reward-json /logs/verifier/reward.json
"""


def ensure_lab_source(dest: str | Path, *, allow_download: bool = True) -> Path:
    dest = Path(dest).expanduser().resolve()
    source_root = dest / LAB_ARCHIVE_ROOT
    if (source_root / "tasks").is_dir():
        return source_root
    if not allow_download:
        raise FileNotFoundError(f"LAB source not found under {dest} and downloads are disabled")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".lab-src-") as tmp:
        archive = Path(tmp) / "lab.tar.gz"
        _download(LAB_ARCHIVE_URL, archive, LAB_ARCHIVE_SHA256)
        _safe_extract(archive, dest)
    if not (source_root / "tasks").is_dir():
        raise RuntimeError(f"extracted archive missing expected root {LAB_ARCHIVE_ROOT}/tasks")
    return source_root


def _download(url: str, out: Path, sha256: str) -> None:
    for attempt in range(1, 4):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=120) as response, out.open("wb") as handle:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError(f"LAB archive checksum mismatch: expected {sha256}, got {digest.hexdigest()}")
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient download/verify error
            out.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"  download attempt {attempt}/3 failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(2 ** (attempt - 1))


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0] != LAB_ARCHIVE_ROOT):
                raise ValueError(f"unsafe archive entry: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported archive entry: {member.name}")
        tar.extractall(dest, filter="data")


def flatten_task_id(source_id: str) -> str:
    parts = PurePosixPath(source_id).parts
    if len(parts) < 2 or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"unexpected LAB task id: {source_id!r}")
    return "__".join(parts)


def iter_source_tasks(source_root: Path) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    tasks_root = source_root / "tasks"
    for task_json in sorted(tasks_root.rglob("task.json")):
        task_dir = task_json.parent
        source_id = task_dir.relative_to(tasks_root).as_posix()
        config = json.loads(task_json.read_text(encoding="utf-8"))
        if not all(config.get(k) for k in ("title", "instructions", "criteria")):
            raise ValueError(f"LAB task {source_id} missing title/instructions/criteria")
        if not (task_dir / "documents").is_dir():
            raise ValueError(f"LAB task {source_id} has no documents/ directory")
        yield source_id, task_dir, config


def _task_toml(config: dict[str, Any], source_id: str, task_name: str, judge_env: dict[str, str]) -> str:
    lines = [
        f'name = "{task_name}"',
        'version = "1.0"',
        "",
        "[metadata]",
        f"lab_task_id = {json.dumps(source_id)}",
        f"title = {json.dumps(config.get('title', ''))}",
        "",
        "[agent]",
        "timeout_sec = 108000",
        "",
        "[verifier]",
        "timeout_sec = 1800",
        "",
        "[environment]",
        "build_timeout_sec = 1800",
        "cpus = 1",
        "memory_mb = 4096",
        "allow_internet = true",  # the verifier calls the judge endpoint
        "",
    ]
    if judge_env:
        lines.append("[verifier.env]")
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in sorted(judge_env.items()))
        lines.append("")
    return "\n".join(lines)


def build_suite(
    source_root: Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    judge_env: dict[str, str] | None = None,
    run_dir: str = "/logs/agent/artifacts/lab-run",
) -> Path:
    """Generate a Harbor task suite under ``out_dir`` from LAB's raw source."""
    if not LAB_VERIFY_SOURCE.is_file():
        raise FileNotFoundError(f"missing verifier template {LAB_VERIFY_SOURCE}")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    judge_env = judge_env or {}
    test_script = _TEST_SCRIPT.format(run_dir=run_dir)
    count = 0
    for source_id, task_dir, config in iter_source_tasks(source_root):
        task_name = flatten_task_id(source_id)
        dst = out / task_name
        (dst / "environment").mkdir(parents=True, exist_ok=True)
        (dst / "tests").mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_dir / "documents", dst / "documents", dirs_exist_ok=True)

        task_json = json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        (dst / "tests" / "task.json").write_text(task_json, encoding="utf-8")
        (dst / "instruction.md").write_text(f"# {config['title']}\n\n{config['instructions']}\n", encoding="utf-8")
        (dst / "task.toml").write_text(_task_toml(config, source_id, task_name, judge_env), encoding="utf-8")
        (dst / "environment" / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        shutil.copyfile(LAB_VERIFY_SOURCE, dst / "tests" / "lab_verify.py")
        test_path = dst / "tests" / "test.sh"
        test_path.write_text(test_script, encoding="utf-8")
        test_path.chmod(test_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        count += 1
        if limit is not None and count >= limit:
            break
    print(f"Generated {count} Harbor tasks under {out}")
    return out


def _non_negative_int(raw: str) -> int:
    """argparse type for ``--limit``: reject negatives so 0 unambiguously means "no tasks"."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return value


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default="./data/lab-source", help="Where LAB source is downloaded/extracted.")
    parser.add_argument("--out-dir", default="./data/lab-harbor-suite", help="Where the Harbor suite is generated.")
    parser.add_argument(
        "--limit", type=_non_negative_int, default=None, help="Generate only the first N tasks (0 generates none)."
    )
    parser.add_argument("--no-download", action="store_true", help="Fail if LAB source is not already present.")
    parser.add_argument("--run-dir", default="/logs/agent/artifacts/lab-run", help="In-container agent output dir.")
    parser.add_argument("--judge-base-url", default=None, help="Bake JUDGE_BASE_URL into each task's [verifier.env].")
    parser.add_argument("--judge-model", default=None, help="Bake JUDGE_MODEL into each task's [verifier.env].")
    parser.add_argument(
        "--judge-api-key", default=None, help="Bake JUDGE_API_KEY into [verifier.env] (writes a secret to disk!)."
    )
    args = parser.parse_args(argv)

    judge_env = {
        env: value
        for env, value in (
            ("JUDGE_BASE_URL", args.judge_base_url),
            ("JUDGE_MODEL", args.judge_model),
            ("JUDGE_API_KEY", args.judge_api_key),
        )
        if value is not None
    }
    source_root = ensure_lab_source(args.source_dir, allow_download=not args.no_download)
    build_suite(source_root, args.out_dir, limit=args.limit, judge_env=judge_env, run_dir=args.run_dir)


if __name__ == "__main__":
    _main()
